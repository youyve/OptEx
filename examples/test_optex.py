import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
import sys

from optex import OptEx

# 配置日志（如果未在 optex.py 中配置，可以在这里配置）
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 定义合成目标函数
def sphere_function(theta: torch.Tensor) -> torch.Tensor:
    """Sphere 函数，常用于优化测试。"""
    return torch.sum(theta.pow(2))

def ackley_function(theta: torch.Tensor) -> torch.Tensor:
    """Ackley 函数，具有多个局部最小值。"""
    d = theta.shape[0]
    sum1 = torch.sum(theta ** 2)
    sum2 = torch.sum(torch.cos(2 * torch.pi * theta))
    term1 = -20 * torch.exp(-0.2 * torch.sqrt(sum1 / d))
    term2 = -torch.exp(sum2 / d)
    return term1 + term2 + 20 + torch.e

def rosenbrock_function(theta: torch.Tensor) -> torch.Tensor:
    """Rosenbrock 函数，亦称香蕉函数。"""
    return torch.sum(100 * (theta[1:] - theta[:-1] ** 2) ** 2 + (1 - theta[:-1]) ** 2)

# 定义 ThetaModel 类
class ThetaModel(nn.Module):
    def __init__(self, d: int, device: str, init_range: tuple = (-5.0, 5.0)):
        """
        初始化 ThetaModel 模型。

        Args:
            d (int): 参数维度。
            device (str): 设备类型。
            init_range (tuple, optional): 参数初始化范围，默认为 (-5.0, 5.0)。
        """
        super(ThetaModel, self).__init__()
        self.theta = nn.Parameter(torch.empty(d, device=device))
        nn.init.uniform_(self.theta, *init_range)

    def forward(self) -> torch.Tensor:
        """前向传播，返回参数向量。"""
        return self.theta

# 优化函数：Vanilla 优化
def optimize_vanilla(
    function,
    d: int,
    T: int,
    lr: float,
    num_runs: int,
    optimizer_type: str = 'SGD',
    init_range: tuple = (-5.0, 5.0),
    max_grad_norm: float = None
):
    """
    使用 Vanilla 优化方法进行优化。

    Args:
        function (callable): 目标函数。
        d (int): 参数维度。
        T (int): 迭代次数。
        lr (float): 学习率。
        num_runs (int): 运行次数。
        optimizer_type (str, optional): 优化器类型，默认为 'SGD'。
        init_range (tuple, optional): 参数初始化范围，默认为 (-5.0, 5.0)。
        max_grad_norm (float, optional): 最大梯度范数，默认为 None。

    Returns:
        list: 每次运行的损失历史。
    """
    loss_histories = []
    # 初始化 OptEx 以检测设备
    devices = OptEx.detect_devices()
    device = torch.device(devices[0])  # 使用第一个可用设备
    for run in range(num_runs):
        theta = nn.Parameter(torch.empty(d, device=device))
        nn.init.uniform_(theta, *init_range)
        if optimizer_type == 'SGD':
            optimizer = optim.SGD([theta], lr=lr)
        elif optimizer_type == 'Adam':
            optimizer = optim.Adam([theta], lr=lr)
        else:
            raise ValueError("不支持的优化器类型。")
        loss_history = []
        for t in range(T):
            optimizer.zero_grad()
            loss = function(theta)
            loss.backward()

            # 梯度裁剪（如果指定）
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_([theta], max_norm=max_grad_norm)

            # 检查非有限梯度
            if not torch.isfinite(theta.grad).all():
                logging.warning("在 Vanilla 优化中遇到非有限梯度。")
                break  # 退出循环

            optimizer.step()

            loss_history.append(loss.item())
            logging.info(f"Vanilla Run {run+1}/{num_runs}, Iteration {t+1}/{T}, Loss: {loss.item()}")
        loss_histories.append(loss_history)
    return loss_histories

# 优化函数：OptEx 优化
def optimize_optex(
    function,
    d: int,
    T: int,
    lr: float,
    devices: list,
    num_runs: int,
    optimizer_type: str = 'SGD',
    init_range: tuple = (-5.0, 5.0),
    max_grad_norm: float = None,
    use_dim_wise_kernel: bool = True
):
    """
    使用 OptEx 优化方法进行优化。

    Args:
        function (callable): 目标函数。
        d (int): 参数维度。
        T (int): 迭代次数。
        lr (float): 学习率。
        devices (list): 设备列表。
        num_runs (int): 运行次数。
        optimizer_type (str, optional): 优化器类型，默认为 'SGD'。
        init_range (tuple, optional): 参数初始化范围，默认为 (-5.0, 5.0)。
        max_grad_norm (float, optional): 最大梯度范数，默认为 None。
        use_dim_wise_kernel (bool, optional): 是否使用维度分解核，默认为 True。

    Returns:
        list: 每次运行的损失历史。
    """
    num_parall = len(devices)
    loss_histories = []
    for run in range(num_runs):
        # 初始化多个模型和优化器（每个设备一个）
        nets = [ThetaModel(d, devices[i], init_range=init_range) for i in range(num_parall)]
        if optimizer_type == 'SGD':
            opts = [optim.SGD(net.parameters(), lr=lr) for net in nets]
        elif optimizer_type == 'Adam':
            opts = [optim.Adam(net.parameters(), lr=lr) for net in nets]
        else:
            raise ValueError("不支持的优化器类型。")

        # 初始化 OptEx
        opt_ex = OptEx(
            num_parall=num_parall,
            max_history=10,
            kernel=None,  # 根据 ARD 和 use_dim_wise_kernel 初始化
            lengthscale=1.0,
            std=0.001,
            ard_num_dims=d if use_dim_wise_kernel else None,
            use_dim_wise_kernel=use_dim_wise_kernel,
            devices=devices
        )

        if not opt_ex.use_dim_wise_kernel:
            opt_ex.reset_activedims(d)

        loss_history = []

        for t in range(T):
            # 运行并行迭代
            opt_ex.run_parallel_iteration(
                nets, opts, function, max_grad_norm=max_grad_norm
            )

            # 每 10 次迭代调优一次 lengthscale 和 active_dims
            if (t + 1) % 10 == 0:
                opt_ex.tune_lengthscale()
                if not opt_ex.use_dim_wise_kernel:
                    opt_ex.tune_activedims(d)

            # 从第一个网络计算当前损失
            theta = nets[0].theta
            loss = function(theta).item()
            loss_history.append(loss)
            logging.info(f"OptEx Run {run+1}/{num_runs}, Iteration {t+1}/{T}, Loss: {loss}")
        loss_histories.append(loss_history)
    return loss_histories

# 主函数
import os

def main():
    # 创建 output 文件夹，如果不存在则创建
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化 OptEx 以检测设备
    devices = OptEx.detect_devices()
    num_runs = 5  # 独立运行次数

    functions = {
        'Sphere': (sphere_function, 0.0),
        'Ackley': (ackley_function, 0.0),
        'Rosenbrock': (rosenbrock_function, 0.0)
    }

    for func_name, (function, f_star) in functions.items():
        logging.info(f"\n运行 {func_name} 函数的优化")

        # 根据函数设置参数
        if func_name == 'Rosenbrock':
            d = int(1e5)
            T = 40
            lr = 0.01
            optimizer_type = 'Adam'
            init_range = (-1.0, 1.0)
            max_grad_norm = 100.0
            use_dim_wise_kernel = True  # 可以根据需要调整
        elif func_name == 'Ackley':
            d = int(1e5)
            T = 40
            lr = 0.01
            optimizer_type = 'Adam'
            init_range = (-1.0, 1.0)
            max_grad_norm = None
            use_dim_wise_kernel = True  # 可以根据需要调整
        else:  # Sphere function
            d = int(1e5)
            T = 40
            lr = 0.01
            optimizer_type = 'Adam'
            init_range = (-1.0, 1.0)
            max_grad_norm = None
            use_dim_wise_kernel = True  # 可以根据需要调整

        # 使用 Vanilla 方法进行优化
        logging.info("运行 Vanilla 优化")
        vanilla_loss_histories = optimize_vanilla(
            function=function,
            d=d,
            T=T,
            lr=lr,
            num_runs=num_runs,
            optimizer_type=optimizer_type,
            init_range=init_range,
            max_grad_norm=max_grad_norm
        )
        vanilla_mean_loss = torch.tensor(vanilla_loss_histories).mean(dim=0).cpu().numpy()
        vanilla_opt_gap = vanilla_mean_loss - f_star

        # 使用 OptEx 方法进行优化
        logging.info("运行 OptEx 优化")
        optex_loss_histories = optimize_optex(
            function=function,
            d=d,
            T=T,
            lr=lr,
            devices=devices,
            num_runs=num_runs,
            optimizer_type=optimizer_type,
            init_range=init_range,
            max_grad_norm=max_grad_norm,
            use_dim_wise_kernel=use_dim_wise_kernel
        )
        optex_mean_loss = torch.tensor(optex_loss_histories).mean(dim=0).cpu().numpy()
        optex_opt_gap = optex_mean_loss - f_star

        epsilon = 1e-20
        vanilla_opt_gap_log = torch.log10(torch.clamp(torch.tensor(vanilla_opt_gap), min=epsilon)).numpy()
        optex_opt_gap_log = torch.log10(torch.clamp(torch.tensor(optex_opt_gap), min=epsilon)).numpy()

        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(vanilla_opt_gap_log) + 1),
            vanilla_opt_gap_log,
            linestyle='--',
            marker='x',
            color='blue',
            label='Vanilla'
        )
        plt.plot(
            range(1, len(optex_opt_gap_log) + 1),
            optex_opt_gap_log,
            linestyle='-',
            marker='o',
            color='orange',
            markerfacecolor='none',
            label='OptEx'
        )
        plt.xlabel('Sequential Iterations T')
        plt.ylabel('Optimality Gap (log scale)')
        plt.title(f'{func_name} Function Optimization')
        plt.legend()

        all_log_gaps = torch.cat([torch.tensor(vanilla_opt_gap_log), torch.tensor(optex_opt_gap_log)])
        all_log_gaps = all_log_gaps[torch.isfinite(all_log_gaps)]
        y_min = int(torch.floor(torch.min(all_log_gaps)).item())
        y_max = int(torch.ceil(torch.max(all_log_gaps)).item())
        y_ticks = range(y_min, y_max + 1, 1)
        plt.yticks(y_ticks, [str(tick) for tick in y_ticks])

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        # 保存图片到 output 文件夹
        output_path = os.path.join(output_dir, f'{func_name}_optimization.png')
        plt.savefig(output_path)
        plt.close()
        logging.info(f"已保存 {func_name} 函数优化的图像为 '{output_path}'。")

if __name__ == "__main__":
    main()