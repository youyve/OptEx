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
    num_parallel = min(1, len(device_list))  # 当前仅支持单机单卡
    optex = initialize_optex(num_parallel=num_parallel, devices=device_list[:num_parallel])

    # 设置模型和优化器
    models, optimizers = setup_parallel_models(num_parallel, device_list[:num_parallel])

    # 使用 OptEx 训练模型
    train(optex, models, optimizers, train_loader, device_list[:num_parallel], num_epochs=5)

    # 评估第一个模型
    evaluate(models[0], device_list[0], test_loader)

if __name__ == "__main__":
    main()
