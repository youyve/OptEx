## Quick start

The following example shows how to use OptEx for optimization. You can find more detailed example code in the examples/ directory.

### Example 1: Optimizing Sphere Functions with OptEx

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
import sys

from optex import OptEx

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 定义合成目标函数
def sphere_function(theta: torch.Tensor) -> torch.Tensor:
    """Sphere 函数，常用于优化测试。"""
    return torch.sum(theta.pow(2))

# 定义 ThetaModel 类
class ThetaModel(nn.Module):
    def __init__(self, d: int, device: str, init_range: tuple = (-5.0, 5.0)):
        super(ThetaModel, self).__init__()
        self.theta = nn.Parameter(torch.empty(d, device=device))
        nn.init.uniform_(self.theta, *init_range)

    def forward(self) -> torch.Tensor:
        return self.theta

# 优化函数：Vanilla 优化
def optimize_vanilla(function, d: int, T: int, lr: float, num_runs: int, optimizer_type: str = 'SGD', init_range: tuple = (-5.0, 5.0), max_grad_norm: float = None
):
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
def optimize_optex(function, d: int, T: int, lr: float, devices: list, num_runs: int, optimizer_type: str = 'SGD', init_range: tuple = (-5.0, 5.0), max_grad_norm: float = None, use_dim_wise_kernel: bool = True):
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
        opt_ex = OptEx(num_parall=num_parall, max_history=10, kernel=None, lengthscale=1.0, std=0.001, ard_num_dims=d if use_dim_wise_kernel else None, use_dim_wise_kernel=use_dim_wise_kernel, devices=devices)

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
    }

    for func_name, (function, f_star) in functions.items():
        logging.info(f"\n运行 {func_name} 函数的优化")
        d = int(1e5)
        T = 40
        lr = 0.01
        optimizer_type = 'Adam'
        init_range = (-1.0, 1.0)
        max_grad_norm = None
        use_dim_wise_kernel = True  # 可以根据需要调整

        # 使用 Vanilla 方法进行优化
        logging.info("运行 Vanilla 优化")
        vanilla_loss_histories = optimize_vanilla(function=function, d=d, T=T, lr=lr, num_runs=num_runs, optimizer_type=optimizer_type, init_range=init_range, max_grad_norm=max_grad_norm)
        vanilla_mean_loss = torch.tensor(vanilla_loss_histories).mean(dim=0).cpu().numpy()
        vanilla_opt_gap = vanilla_mean_loss - f_star

        # 使用 OptEx 方法进行优化
        logging.info("运行 OptEx 优化")
        optex_loss_histories = optimize_optex(function=function, d=d, T=T, lr=lr, devices=devices, num_runs=num_runs, optimizer_type=optimizer_type, init_range=init_range, max_grad_norm=max_grad_norm, use_dim_wise_kernel=use_dim_wise_kernel)
        optex_mean_loss = torch.tensor(optex_loss_histories).mean(dim=0).cpu().numpy()
        optex_opt_gap = optex_mean_loss - f_star

        epsilon = 1e-20
        vanilla_opt_gap_log = torch.log10(torch.clamp(torch.tensor(vanilla_opt_gap), min=epsilon)).numpy()
        optex_opt_gap_log = torch.log10(torch.clamp(torch.tensor(optex_opt_gap), min=epsilon)).numpy()

        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(vanilla_opt_gap_log) + 1), vanilla_opt_gap_log, linestyle='--', marker='x', color='blue', label='Vanilla')
        plt.plot(range(1, len(optex_opt_gap_log) + 1), optex_opt_gap_log, linestyle='-', marker='o', color='orange', label='OptEx')
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
```



### Example 2: Optimizing a MNIST Dataset with OptEx

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from optex.optex import OptEx  # 导入 OptEx
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self):
        # 将所有参数连接成一个向量 (theta)
        params = []
        for param in self.parameters():
            params.append(param.view(-1))
        theta = torch.cat(params)
        return theta

def get_data_loaders(batch_size=64):
    # 数据预处理，包括转为张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载 MNIST 训练数据集和测试数据集
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader

def objective_function(model, device, data, target):
    # 将模型和数据移至设备
    model.to(device)
    data, target = data.to(device), target.to(device)
    output = model.flatten(data)
    output = model.fc1(output)
    output = model.relu(output)
    output = model.fc2(output)
    # 计算交叉熵损失
    loss = F.cross_entropy(output, target)
    return loss

def objective_function_theta(theta, models, device_list, data, target):
    # 用 theta 更新模型参数
    idx = 0
    for param in models[0].parameters():
        param_length = param.numel()
        param.data = theta[idx:idx + param_length].view_as(param).data
        idx += param_length
    return objective_function(models[0], device_list[0], data, target)

def initialize_optex(num_parallel=1, devices=None):
    if devices is None:
        # 自动检测设备
        device_list = OptEx.detect_devices()
    else:
        device_list = devices

    # 初始化 OptEx 优化器
    optex = OptEx(
        num_parall=num_parallel,
        max_history=100,
        kernel=None,  # 使用默认的 Matern kernel
        lengthscale=1.0,
        std=0.001,
        ard_num_dims=None,  # 不使用 ARD
        use_dim_wise_kernel=True,
        devices=device_list
    )
    return optex

def setup_parallel_models(num_parallel, device_list):
    models = []
    optimizers = []
    for i in range(num_parallel):
        # 为每个设备创建一个模型和优化器
        model = SimpleNN().to(device_list[i])
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        models.append(model)
        optimizers.append(optimizer)
    return models, optimizers

def train(optex, models, optimizers, train_loader, device_list, num_epochs=5):
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(train_loader):
            # 定义一个用于目标函数的闭包
            def closure():
                return objective_function(models[0], device_list[0], data, target)
            
            # 执行代理更新
            optex.run_proxy_update(
                nets=models, 
                opts=optimizers, 
                function=lambda theta: objective_function_theta(theta, models, device_list, data, target), 
                num_iterations=1, 
                max_grad_norm=5.0
            )
            
            if batch_idx % 100 == 0:
                logging.info(f"Batch {batch_idx}: 已完成")

def evaluate(model, device, test_loader):
    # 模型评估
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.flatten(data)
            output = model.fc1(output)
            output = model.relu(output)
            output = model.fc2(output)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    # 计算准确率
    accuracy = 100. * correct / total
    logging.info(f"测试准确率: {accuracy:.2f}%")

def main():
    # 初始化设备和 OptEx
    train_loader, test_loader = get_data_loaders(batch_size=64)
    device_list = OptEx.detect_devices()
    num_parallel = min(1, len(device_list))  # 当前仅支持单机单卡，可自动识别GPU、NPU、MPS和CPU
    optex = initialize_optex(num_parallel=num_parallel, devices=device_list[:num_parallel])

    # 设置模型和优化器
    models, optimizers = setup_parallel_models(num_parallel, device_list[:num_parallel])

    # 使用 OptEx 训练模型
    train(optex, models, optimizers, train_loader, device_list[:num_parallel], num_epochs=5)

    # 评估第一个模型
    evaluate(models[0], device_list[0], test_loader)

if __name__ == "__main__":
    main()
```