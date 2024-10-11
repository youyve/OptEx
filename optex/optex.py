import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gpytorch.kernels import MaternKernel
from gpytorch.constraints import Positive
import logging
import sys
import concurrent.futures

# 尝试导入 torch_npu，如果不可用则跳过
try:
    import torch_npu
except ImportError:
    torch_npu = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class OptEx:
    def __init__(
        self,
        num_parall: int,
        max_history: int = 50,
        kernel: nn.Module = None,
        lengthscale: float = 1.0,
        std: float = 0.001,
        ard_num_dims: int = None,
        use_dim_wise_kernel: bool = True,
        devices: list = None
    ):
        """
        初始化 OptEx 优化加速器。

        Args:
            num_parall (int): 并行优化的数量。
            max_history (int, optional): 历史记录的最大长度。默认为 50。
            kernel (nn.Module, optional): 使用的核函数，默认为 None。
            lengthscale (float, optional): 初始长度尺度，默认为 1.0。
            std (float, optional): 标准差，默认为 0.001。
            ard_num_dims (int, optional): ARD 的维度数量，默认为 None。
            use_dim_wise_kernel (bool, optional): 是否使用维度分解核，默认为 True。
            devices (list, optional): 可用的设备列表。如果为 None，将自动检测设备
        """
        self.num_parall = num_parall
        self.max_history = max_history
        self.active_dims = None
        self.std = std
        self.use_dim_wise_kernel = use_dim_wise_kernel

        # 设备检测
        if devices is None:
            self.devices = self.detect_devices()
        else:
            self.devices = devices

        self.device = torch.device(self.devices[0])  # 默认使用第一个设备

        if kernel is None:
            if self.use_dim_wise_kernel and ard_num_dims:
                # 使用 ARD，每个维度单独的长度尺度
                self.kernel = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims).to(self.device)
            else:
                self.kernel = MaternKernel(nu=2.5).to(self.device)
        else:
            self.kernel = kernel.to(self.device)

        self.kernel.raw_lengthscale_constraint = Positive(transform=None)
        self.reset_lengthscale(lengthscale)
        self.param_history = []  # 参数历史记录
        self.grad_history = []   # 梯度历史记录

    @staticmethod
    def detect_devices() -> list:
        """
        检测可用的计算设备。

        Returns:
            list: 可用设备的列表。
        """
        devices = []
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            devices.append('mps')
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            devices += [f'cuda:{i}' for i in range(num_gpus)]
        if torch_npu is not None and torch_npu.npu.is_available():
            num_npus = torch_npu.npu.device_count()
            devices += [f'npu:{i}' for i in range(num_npus)]
        if not devices:
            devices.append('cpu')
        return devices

    def reset_lengthscale(self, lengthscale: float):
        """重置核函数的长度尺度。"""
        with torch.no_grad():
            lengthscale_tensor = torch.tensor(lengthscale, dtype=torch.float32, device=self.device)
            if hasattr(self.kernel, 'raw_lengthscale'):
                self.kernel.raw_lengthscale.data.copy_(lengthscale_tensor)
            else:
                self.kernel.lengthscale.data.copy_(lengthscale_tensor)

    def reset_activedims(self, d: int, active_dims: torch.Tensor = None, effective_dim: int = 5000):
        """重置活动维度。"""
        if not self.use_dim_wise_kernel:
            if active_dims is None:
                effective_dim = min(d, effective_dim)
                self.active_dims = torch.randperm(d, device=self.device)[:effective_dim]
            else:
                self.active_dims = active_dims.to(self.device)
        else:
            self.active_dims = None

    def tune_activedims(self, d: int, choices: list = None, effective_dim: int = 5000):
        """调整活动维度以优化性能。"""
        if choices is None:
            choices = [500, 1000, 2000, 5000, 10000, d]
        if self.use_dim_wise_kernel:
            return  # 使用维度分解核时无需调整活动维度

        if len(self.param_history) < 4:
            return

        errors = []
        for eff_dim in choices:
            self.reset_activedims(d, effective_dim=eff_dim)
            xs_train, xs_test, ys_train, ys_test = self._prepare_training_data()

            k_vec = self.kernel(xs_test, xs_train)
            K_mat = self.kernel(xs_train, xs_train)
            K_mat += torch.eye(K_mat.size(-1), device=self.device) * 1e-6  # 数值稳定性

            try:
                K_mat_dense = K_mat.to_dense()
            except NotImplementedError:
                errors.append(float('inf'))
                continue

            if not torch.isfinite(K_mat_dense).all():
                errors.append(float('inf'))
                continue

            K_mat_inv = torch.linalg.pinv(K_mat_dense)
            grad_pred = k_vec.to_dense() @ K_mat_inv @ ys_train

            error = 1 - F.cosine_similarity(
                grad_pred.view(len(xs_test), -1),
                ys_test.view(len(xs_test), -1),
                dim=-1
            ).mean().item()
            errors.append(error)

        if not errors:
            return

        # 找到误差最小的索引
        idx = torch.argmin(torch.tensor(errors)).item()
        optimal_dim = choices[idx]
        self.reset_activedims(d, effective_dim=optimal_dim)

    def tune_lengthscale(self, choices: list = None):
        """调整核函数的长度尺度以优化性能。"""
        if choices is None:
            choices = [1e-12, 1e-6, 1e-4, 1e-2, 1e-1, 1, 1e1, 1e2, 1e4, 1e6, 1e12]
        if len(self.param_history) < 4:
            return

        xs_train, xs_test, ys_train, ys_test = self._prepare_training_data()

        errors = []
        for ls in choices:
            self.reset_lengthscale(ls)

            k_vec = self.kernel(xs_test, xs_train)
            K_mat = self.kernel(xs_train, xs_train)

            K_mat += torch.eye(K_mat.size(-1), device=self.device) * 1e-6  # 数值稳定性

            try:
                K_mat_dense = K_mat.to_dense()
            except NotImplementedError:
                errors.append(float('inf'))
                continue

            if not torch.isfinite(K_mat_dense).all():
                errors.append(float('inf'))
                continue

            K_mat_inv = torch.linalg.pinv(K_mat_dense)
            grad_pred = k_vec.to_dense() @ K_mat_inv @ ys_train

            error = 1 - F.cosine_similarity(
                grad_pred.view(len(xs_test), -1),
                ys_test.view(len(xs_test), -1),
                dim=-1
            ).mean().item()
            errors.append(error)

        if not errors:
            return

        # 找到误差最小的索引
        idx = torch.argmin(torch.tensor(errors)).item()
        optimal_ls = choices[idx]
        self.reset_lengthscale(optimal_ls)

    def _prepare_training_data(self):
        """准备训练和测试数据。"""
        num_history = len(self.param_history)
        num_train = int(0.8 * num_history)
        indices = torch.randperm(num_history, device=self.device)
        train_inds = indices[:num_train]
        test_inds = indices[num_train:]

        xs_train = torch.stack([self.param_history[i] for i in train_inds], dim=0).to(self.device)
        xs_test = torch.stack([self.param_history[i] for i in test_inds], dim=0).to(self.device)
        ys_train = torch.stack([self.grad_history[i] for i in train_inds], dim=0).to(self.device)
        ys_test = torch.stack([self.grad_history[i] for i in test_inds], dim=0).to(self.device)

        if not self.use_dim_wise_kernel and self.active_dims is not None:
            xs_train = xs_train[:, self.active_dims]
            xs_test = xs_test[:, self.active_dims]

        # 归一化
        xs_train = F.normalize(xs_train, p=2, dim=1, eps=1e-12)
        xs_test = F.normalize(xs_test, p=2, dim=1, eps=1e-12)

        return xs_train, xs_test, ys_train, ys_test

    def update_history(self, param_list: list, grad_list: list):
        """更新参数和梯度的历史记录。"""
        for param, grad in zip(param_list, grad_list):
            if torch.isfinite(param).all() and torch.isfinite(grad).all():
                self.param_history.append(param.detach().to(self.device))
                self.grad_history.append(grad.detach().to(self.device))

        # 保留最后 max_history 条记录
        if len(self.param_history) > self.max_history:
            self.param_history = self.param_history[-self.max_history:]
            self.grad_history = self.grad_history[-self.max_history:]

    def get_proxy_grad_func(self):
        """生成代理梯度函数。"""
        if len(self.param_history) == 0 or len(self.grad_history) == 0:
            return lambda x: torch.zeros_like(x)  # 返回一个虚拟函数

        xs = torch.stack(self.param_history, dim=0).to(self.device)
        ys = torch.stack(self.grad_history, dim=0).to(self.device)

        if not self.use_dim_wise_kernel and self.active_dims is not None:
            xs = xs[:, self.active_dims]

        xs = F.normalize(xs, p=2, dim=1, eps=1e-12)

        K_mat = self.kernel(xs, xs)
        K_mat += torch.eye(K_mat.size(-1), device=self.device) * 1e-6  # 数值稳定性
        try:
            K_mat_dense = K_mat.to_dense()
        except NotImplementedError:
            return lambda x: torch.zeros_like(x)  # 返回一个虚拟函数

        K_mat_inv = torch.linalg.pinv(K_mat_dense)

        def proxy_grad_func(x: torch.Tensor) -> torch.Tensor:
            x = x.view(1, -1).to(self.device)
            if not self.use_dim_wise_kernel and self.active_dims is not None:
                x = x[:, self.active_dims]
            x = F.normalize(x, p=2, dim=1, eps=1e-12)

            k_vec = self.kernel(x, xs)

            try:
                k_vec_dense = k_vec.to_dense()
            except NotImplementedError:
                return torch.zeros(x.size(0), ys.size(1), device=self.device)

            grad_pred = k_vec_dense @ K_mat_inv @ ys
            return grad_pred.squeeze(0)

        return proxy_grad_func

    def run_iteration(self, net: nn.Module, opt: optim.Optimizer, function, device: str, max_grad_norm: float = None):
        """
        运行单次优化迭代。

        Args:
            net (nn.Module): 优化的模型。
            opt (optim.Optimizer): 优化器。
            function (callable): 目标函数。
            device (str): 设备类型。
            max_grad_norm (float, optional): 最大梯度范数，默认为 None。

        Returns:
            tuple: (state_dict, optimizer_state_dict, grad_vector, loss_value) 或 None
        """
        device = torch.device(device)
        net.to(device)
        opt.zero_grad()
        theta = net.forward()
        loss = function(theta)
        loss.backward()

        # 梯度裁剪（如果指定）
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=max_grad_norm)

        finite_grads = True
        for p in net.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                finite_grads = False
                break

        if not finite_grads:
            return None  # 跳过此迭代

        opt.step()

        grad_vector = nn.utils.parameters_to_vector(
            [p.grad for p in net.parameters() if p.grad is not None]
        ).detach().clone()

        state = net.state_dict()
        opt_state = opt.state_dict()
        grad = grad_vector
        loss_value = loss.item()
        return (state, opt_state, grad, loss_value)

    def run_parallel_iteration(
        self,
        nets: list,
        opts: list,
        function,
        max_grad_norm: float = None
    ):
        """
        运行并行迭代。

        Args:
            nets (list): 模型列表。
            opts (list): 优化器列表。
            function (callable): 目标函数。
            max_grad_norm (float, optional): 最大梯度范数，默认为 None。

        Returns:
            None
        """
        updated_states = []
        updated_opts = []
        updated_grads = []
        losses = []

        def run_on_device(i: int):
            return self.run_iteration(nets[i], opts[i], function, self.devices[i], max_grad_norm)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parall) as executor:
            futures = [executor.submit(run_on_device, i) for i in range(self.num_parall)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                state, opt_state, grad, loss_value = result
                updated_states.append(state)
                updated_opts.append(opt_state)
                updated_grads.append(grad)
                losses.append(loss_value)

        if not updated_states:
            return

        param_vectors = [nn.utils.parameters_to_vector(state.values()) for state in updated_states]
        self.update_history(param_vectors, updated_grads)

        if not losses:
            return

        optimal_idx = torch.argmin(torch.tensor(losses)).item()
        optimal_net_state = updated_states[optimal_idx]
        optimal_opt_state = updated_opts[optimal_idx]

        # 将最佳状态加载到所有模型和优化器中
        for net, opt_, device in zip(nets, opts, self.devices):
            device = torch.device(device)
            net_state_on_device = {k: v.to(device) for k, v in optimal_net_state.items()}
            net.load_state_dict(net_state_on_device)
            net.to(device)

            opt_state_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in optimal_opt_state.items()}
            opt_.load_state_dict(opt_state_on_device)

    def run_proxy_update(self, nets: list, opts: list, function, num_iterations: int = 1, max_grad_norm: float = None):
        """
        运行代理梯度更新。

        Args:
            nets (list): 模型列表。
            opts (list): 优化器列表。
            function (callable): 目标函数。
            num_iterations (int, optional): 代理更新的次数，默认为 1。
            max_grad_norm (float, optional): 最大梯度范数，默认为 None。

        Returns:
            None
        """
        for _ in range(num_iterations):
            self.run_parallel_iteration(
                nets, opts, function, max_grad_norm
            )

            # 每 10 次迭代调优一次 lengthscale 和 active_dims
            if (len(self.param_history) % 10) == 0:
                self.tune_lengthscale()
                if not self.use_dim_wise_kernel:
                    d = nets[0].theta.numel()
                    self.tune_activedims(d)
