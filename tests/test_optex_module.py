import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from optex.optex import OptEx

# 定义与 OptEx 兼容的模型
class ThetaModel(nn.Module):
    def __init__(self, d: int, device: str, init_range: tuple = (-5.0, 5.0)):
        super(ThetaModel, self).__init__()
        self.theta = nn.Parameter(torch.empty(d, device=device))
        nn.init.uniform_(self.theta, *init_range)

    def forward(self) -> torch.Tensor:
        return self.theta

class TestOptExModule(unittest.TestCase):
    def setUp(self):
        # 初始化 OptEx 实例，设置 num_parall=1 以确保至少有一个设备
        self.opt_ex = OptEx(
            num_parall=1,
            max_history=10,
            lengthscale=1.0,
            std=0.001,
            ard_num_dims=100,
            use_dim_wise_kernel=True
        )
        self.d = 100  # 参数维度
        self.num_runs = 3
        self.T = 20
        self.lr = 0.01

    def test_detect_devices(self):
        devices = self.opt_ex.detect_devices()
        self.assertIsInstance(devices, list)
        self.assertGreaterEqual(len(devices), 1)
        for device in devices:
            self.assertIsInstance(device, str)

    def test_update_history(self):
        param = torch.randn(self.d, device=self.opt_ex.device)
        grad = torch.randn(self.d, device=self.opt_ex.device)
        self.opt_ex.update_history([param], [grad])
        self.assertEqual(len(self.opt_ex.param_history), 1)
        self.assertEqual(len(self.opt_ex.grad_history), 1)

        # 添加超过 max_history 的历史记录
        for _ in range(15):
            param = torch.randn(self.d, device=self.opt_ex.device)
            grad = torch.randn(self.d, device=self.opt_ex.device)
            self.opt_ex.update_history([param], [grad])
        self.assertEqual(len(self.opt_ex.param_history), self.opt_ex.max_history)
        self.assertEqual(len(self.opt_ex.grad_history), self.opt_ex.max_history)

    def test_run_iteration(self):
        # 定义一个简单的模型和优化器
        model = ThetaModel(d=self.d, device=self.opt_ex.device)
        optimizer = optim.SGD(model.parameters(), lr=self.lr)

        # 定义一个简单的损失函数
        def loss_fn(theta):
            return torch.sum(theta.pow(2))

        # 运行单次迭代
        result = self.opt_ex.run_iteration(model, optimizer, loss_fn, self.opt_ex.devices[0])
        self.assertIsNotNone(result)
        state, opt_state, grad, loss_value = result
        self.assertIsInstance(state, dict)
        self.assertIsInstance(opt_state, dict)
        self.assertIsInstance(grad, torch.Tensor)
        self.assertIsInstance(loss_value, float)

    def test_run_parallel_iteration(self):
        # 定义一个简单的模型和优化器
        model = ThetaModel(d=self.d, device=self.opt_ex.device)
        optimizer = optim.SGD(model.parameters(), lr=self.lr)
        nets = [model]
        opts = [optimizer]

        # 定义一个简单的损失函数
        def loss_fn(theta):
            return torch.sum(theta.pow(2))

        # 运行并行迭代
        self.opt_ex.run_parallel_iteration(nets, opts, loss_fn)

        # 检查历史记录是否更新
        self.assertGreaterEqual(len(self.opt_ex.param_history), 1)
        self.assertGreaterEqual(len(self.opt_ex.grad_history), 1)

    def test_get_proxy_grad_func(self):
        # 添加一些历史记录
        for _ in range(5):
            param = torch.randn(self.d, device=self.opt_ex.device)
            grad = torch.randn(self.d, device=self.opt_ex.device)
            self.opt_ex.update_history([param], [grad])

        proxy_grad_func = self.opt_ex.get_proxy_grad_func()
        self.assertTrue(callable(proxy_grad_func))

        # 测试代理梯度函数
        test_input = torch.randn(self.d, device=self.opt_ex.device)
        grad_pred = proxy_grad_func(test_input)
        self.assertIsInstance(grad_pred, torch.Tensor)
        self.assertEqual(grad_pred.shape, (self.d,))

if __name__ == '__main__':
    unittest.main()
