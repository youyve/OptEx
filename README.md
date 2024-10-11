# OptEx

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-brightgreen.svg)](https://pytorch.org/)
[![GPyTorch](https://img.shields.io/badge/GPyTorch-1.13-blue.svg)](https://gpytorch.ai/)

[toc]

FOO算法通常需要进行许多顺序迭代才能达到收敛，尤其是在处理复杂优化问题时，这些顺序迭代可能会导致计算效率低下。**OptEx（Optimization Expedited with approximately parallelized iterations）**是一种新提出的框架，旨在通过近似并行化FOO中的顺序迭代来提高优化效率。该项目是一个基于 PyTorch 和 GPyTorch 实现的 OptEx 一阶优化加速框架，OptEx 支持多设备（GPU、NPU、MPS 和 CPU）并行迭代。

## 特性

- **自动相关确定（ARD）**：高效处理多维数据，每个维度单独的长度尺度。
- **长度尺度调整**：自动调整核函数的长度尺度以优化性能。
- **并行迭代**：支持多设备（GPU、NPU、MPS 和 CPU）并行运行优化过程。
- **历史记录管理**：保留参数和梯度的历史记录以供优化使用。
- **设备自动检测**：自动检测并使用可用的最佳计算设备。
- **可扩展性强**：可与任何模型和损失函数集成。

## 项目说明

```bash
OptEx/
├── README.md
├── setup.py
├── requirements.txt
├── .gitignore
├── optex/
│   ├── __init__.py
│   ├── optex.py
│   └── optex_debug.py
├── examples/
│   ├── test_optex.py
│   └── example_usage.py
└── tests/
    ├── __init__.py
    ├── test_optex_module.py
    └── test_examples.py

```

- **optex/**: 主模块目录，包含 `OptEx` 类及其相关代码。
  - `optex/optex.py`: `OptEx` 框架的核心文件，包含 `OptEx` 类及其相关方法。
  - `optex/optex_debug.py`: `OptEx` 的调试版本，包含一些调试信息和日志。
- **examples/**: 提供使用 `OptEx` 的示例脚本。
  - `examples/test_optex.py`: 基本的测试脚本，演示如何使用 `OptEx` 进行优化。使用三个常见的优化目标函数：Sphere、Ackley 和 Rosenbrock。
  - `examples/minst.py`: 是一个使用 `OptEx` 优化 MNIST 数据集的示例脚本。
- **tests/**: 包含单元测试。
  - `tests/test_optex_module.py`: 测试 `OptEx` 模块的核心功能，如设备检测、历史记录管理和优化流程。
  - `tests/test_examples.py`: 测试 `examples/` 目录下的示例脚本，确保它们能够正常运行。


## 安装
**本项目支持在 NVIDIA GPU、Ascend NPU、Apple MPS 和 CPU 上运行。**

### 先决条件

- Python 3.7+
- [PyTorch](https://pytorch.org/) 2.1.0+
- [GPyTorch](https://gpytorch.ai/) 1.13+
- 可选：
  - [torch_npu](https://gitee.com/ascend/pytorch/)（如果使用昇腾 NPU 设备，请根据Pytorch和CANN的版本安装对应的 `torch_npu` ）
  - [torchvision](https://github.com/pytorch/vision#installation)（运行优化 MNIST 数据集时会用到，请根据Pytorch的版本安装对应的torchvision）


### 安装步骤

1. **克隆仓库**

   ```bash
   git clone https://github.com/youyve/OptEx.git
   cd OptEx
   ```

3. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

4. **安装OptEx**

   ```bash
   python setup.py install
   ```

5. **运行测试单元**

   ```bash
   # 测试 OptEx 模块
   python tests/test_optex_module.py
   
   # 测试 examples 目录下的示例脚本（用时较长不推荐）
   python tests/test_examples.py
   
   # 运行所有测试（用时较长不推荐）
   python -m unittest discover -s tests
   ```


## 快速开始

以下示例展示了如何使用 OptEx 进行优化。你可以在 examples/ 目录中找到更详细的示例代码。

### 示例1：使用 `OptEx` 优化 Sphere 函数

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

### 示例2：使用 `OptEx` 优化 MNIST 数据集

运行 `examples/minst.py` 需要安装与torch版本相对应的torchvision

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



## API 文档

### 类：`OptEx`

`OptEx` 是优化加速器的核心类，提供了一系列方法来管理和执行优化过程。以下是 `OptEx` 类的详细 API 文档。

#### `__init__(self, num_parall: int, max_history: int = 50, kernel: nn.Module = None, lengthscale: float = 1.0, std: float = 0.001, ard_num_dims: int = None, use_dim_wise_kernel: bool = True, devices: list = None)`

初始化 `OptEx` 优化加速器。

**参数:**

- `num_parall (int)`: 并行优化的数量。
- `max_history (int, optional)`: 历史记录的最大长度。默认为 `50`。
- `kernel (nn.Module, optional)`: 使用的核函数，默认为 `None`。如果为 `None`，将使用默认的 `MaternKernel`。
- `lengthscale (float, optional)`: 初始长度尺度，默认为 `1.0`。
- `std (float, optional)`: 标准差，默认为 `0.001`。
- `ard_num_dims (int, optional)`: 自动相关确定（ARD）的维度数量，默认为 `None`。
- `use_dim_wise_kernel (bool, optional)`: 是否使用维度分解核，默认为 `True`。
- `devices (list, optional)`: 可用设备列表。如果为 `None`，将自动检测设备。

**示例:**

```python
from optex import OptEx

opt_ex = OptEx(
    num_parall=2,
    max_history=100,
    lengthscale=0.5,
    use_dim_wise_kernel=False
)
```



#### `detect_devices() -> list`

静态方法，用于检测可用的计算设备（GPU、NPU、MPS 或 CPU）。

**返回:**

- `list`: 可用设备的列表，例如 `['npu:0', 'npu:1']`。

**示例:**

```python
devices = OptEx.detect_devices()
print(devices)  # 输出: ['npu:0', 'npu:1']
```



####`reset_lengthscale(self, lengthscale: float)`

重置核函数的长度尺度。

**参数:**

- `lengthscale (float)`: 新的长度尺度值。

**示例:**

```python
opt_ex.reset_lengthscale(2.0)
```



####`reset_activedims(self, d: int, active_dims: torch.Tensor = None, effective_dim: int = 5000)`

重置活动维度，用于优化核函数的性能。

**参数:**

- `d (int)`: 总维度数。
- `active_dims (torch.Tensor, optional)`: 指定的活动维度。如果为 `None`，将随机选择 `effective_dim` 个维度。
- `effective_dim (int, optional)`: 有效维度数量，默认为 `5000`。

**示例:**

```python
# 随机选择 3000 个有效维度
opt_ex.reset_activedims(d=10000, effective_dim=3000)

# 指定特定的活动维度
active_dims = torch.tensor([0, 1, 2, 3, 4])
opt_ex.reset_activedims(d=5, active_dims=active_dims)
```



####`tune_activedims(self, d: int, choices: list = None, effective_dim: int = 5000)`

调整活动维度以优化性能。

**参数:**

- `d (int)`: 总维度数。
- `choices (list, optional)`: 有效维度的候选值列表。默认为 `[500, 1000, 2000, 5000, 10000, d]`。
- `effective_dim (int, optional)`: 默认有效维度数量，默认为 `5000`。

**示例:**

```python
# 使用默认候选值调整活动维度
opt_ex.tune_activedims(d=10000)

# 使用自定义候选值调整活动维度
opt_ex.tune_activedims(d=10000, choices=[1000, 2000, 3000])
```



#### `tune_lengthscale(self, choices: list = None)`

调整核函数的长度尺度以优化性能。

**参数:**

- `choices (list, optional)`: 长度尺度的候选值列表。默认为 `[1e-12, 1e-6, 1e-4, 1e-2, 1e-1, 1, 1e1, 1e2, 1e4, 1e6, 1e12]`。

**示例:**

```python
# 使用默认候选值调整长度尺度
opt_ex.tune_lengthscale()

# 使用自定义候选值调整长度尺度
opt_ex.tune_lengthscale(choices=[0.1, 1.0, 10.0])
```



#### `update_history(self, param_list: list, grad_list: list)`

更新参数和梯度的历史记录。

**参数:**

- `param_list (list)`: 参数向量列表，每个元素为一个 `torch.Tensor`。
- `grad_list (list)`: 梯度向量列表，每个元素为一个 `torch.Tensor`。

**示例:**

```python
# 假设有新的参数和梯度
new_params = [torch.randn(100, device='cuda:0')]
new_grads = [torch.randn(100, device='cuda:0')]

opt_ex.update_history(new_params, new_grads)
```



#### `get_proxy_grad_func(self) -> callable`

生成代理梯度函数，用于预测梯度。

**返回:**

- `callable`: 代理梯度函数，接受一个参数向量 `x` 并返回预测的梯度向量。

**示例:**

```python
proxy_grad = opt_ex.get_proxy_grad_func()
theta = torch.randn(100, device='cuda:0')
predicted_grad = proxy_grad(theta)
```



#### `run_iteration(self, net: nn.Module, opt: optim.Optimizer, function: callable, device: str, max_grad_norm: float = None) -> tuple or None`

运行单次优化迭代。

**参数:**

- `net (nn.Module)`: 优化的模型。
- `opt (optim.Optimizer)`: 优化器。
- `function (callable)`: 目标函数，接收模型并返回损失。
- `device (str)`: 设备类型，例如 `'cuda:0'`。
- `max_grad_norm (float, optional)`: 最大梯度范数，用于梯度裁剪。默认为 `None`。

**返回:**

- `tuple` 或 `None`: 返回 `(state_dict, optimizer_state_dict, grad_vector, loss_value)`，如果遇到非有限梯度则返回 `None`。

**示例:**

```python
result = opt_ex.run_iteration(model, optimizer, sphere_function, 'cuda:0', max_grad_norm=1.0)
if result:
    state, opt_state, grad, loss = result
    print(f"Loss: {loss}")
else:
    print("迭代被跳过 due to non-finite gradients.")
```



#### `run_parallel_iteration(self, nets: list, opts: list, function: callable, max_grad_norm: float = None) -> None`

运行并行迭代，支持多设备优化。

**参数:**

- `nets (list)`: 模型列表，每个模型对应一个设备。
- `opts (list)`: 优化器列表，每个优化器对应一个模型。
- `function (callable)`: 目标函数，接收模型并返回损失。
- `max_grad_norm (float, optional)`: 最大梯度范数，用于梯度裁剪。默认为 `None`。

**示例:**

```python
nets = [model1, model2]
opts = [optimizer1, optimizer2]
opt_ex.run_parallel_iteration(nets, opts, sphere_function, max_grad_norm=1.0)
```



#### `run_proxy_update(self, nets: list, opts: list, function: callable, num_iterations: int = 1, max_grad_norm: float = None) -> None`

运行代理梯度更新，用于基于历史数据优化梯度预测。

**参数:**

- `nets (list)`: 模型列表，每个模型对应一个设备。
- `opts (list)`: 优化器列表，每个优化器对应一个模型。
- `function (callable)`: 目标函数，接收模型并返回损失。
- `num_iterations (int, optional)`: 代理更新的次数，默认为 `1`。
- `max_grad_norm (float, optional)`: 最大梯度范数，用于梯度裁剪。默认为 `None`。

**示例:**

```python
opt_ex.run_proxy_update(nets, opts, sphere_function, num_iterations=5, max_grad_norm=1.0)
```



### 示例类：`ThetaModel`

`ThetaModel` 是一个与 `OptEx` 兼容的简单模型，用于示例和测试。

```python
import torch
import torch.nn as nn

class ThetaModel(nn.Module):
    def __init__(self, d: int, device: str, init_range: tuple = (-5.0, 5.0)):
        super(ThetaModel, self).__init__()
        self.theta = nn.Parameter(torch.empty(d, device=device))
        nn.init.uniform_(self.theta, *init_range)

    def forward(self) -> torch.Tensor:
        return self.theta
```

**方法:**

- `forward(self) -> torch.Tensor`: 返回参数向量 `theta`，不需要输入参数。

**示例:**

```python
model = ThetaModel(d=100, device='cuda:0')
theta = model()
print(theta.shape)  # 输出: torch.Size([100])
```