# OptEx: Expediting First-Order Optimization with Approximately Parallelized Iterations

This repo contains the official implementation for the work "OptEx: Expediting First-Order Optimization with Approximately Parallelized Iterations". See more details in our [paper](https://arxiv.org/abs/2402.11427)

> First-order optimization (FOO) algorithms are pivotal in numerous computational domains, such as reinforcement learning and deep learning. However, their application to complex tasks often entails significant optimization inefficiency due to their need of many sequential iterations for convergence. In response, we introduce *first-order optimization expedited with approximately parallelized iterations* (OptEx), the first general framework that enhances the optimization efficiency of FOO by leveraging parallel computing to directly mitigate its requirement of many sequential iterations for convergence. To achieve this, OptEx utilizes a kernelized gradient estimation that is based on the history of evaluated gradients to predict the gradients required by the next few sequential iterations in FOO, which helps to break the inherent iterative dependency and hence enables the approximate parallelization of iterations in FOO. We further establish theoretical guarantees for the estimation error of our kernelized gradient estimation and the iteration complexity of SGD-based OptEx, confirming that the estimation error diminishes to zero as the history of gradients accumulates and that our SGD-based OptEx enjoys an effective acceleration rate of Θ(*√* *N*) over standard SGD given parallelism of *N*, in terms of the sequential iterations required for convergence. Finally, we provide extensive empirical studies, including synthetic functions, reinforcement learning tasks, and neural network training on various datasets, to underscore the substantial efficiency improvements achieved by our OptEx in practice.



## Project Structure

**This project supports running on NVIDIA GPUs, Ascend NPUs, Apple MPS and CPUs. It also includes implementations of OptEx based on JAX and Pytorch**

```bash
.
├── README.md
├── setup.py
├── requirements.txt
├── .gitignore
├── optex/
│   ├── __init__.py
│   ├── optex.py  // implementation of OptEx with Pytorch
│   └── optex_debug.py  // debug version of OptEx
├── optex_jax/OptEx_JAX.py  //  implementation of OptEx with JAX
├── examples/
│   ├── test_optex.py  // script for testing Sphere, Ackley and Rosenbrock
│   └── minist.py  // script for optimizing MNIST datasets with OptEx
└── tests/
    ├── __init__.py
    ├── test_optex_module.py  // test the core functions of the OptEx module
    └── test_examples.py  // test the scripts in the examples/ directory
```



## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/) 2.1.0+
- [GPyTorch](https://gpytorch.ai/) 1.13+
- 可选：
  - [torch_npu](https://gitee.com/ascend/pytorch/)（If you are using an Ascend NPU device, install the corresponding torch_npu based on the Pytorch and CANN versions）
  - [torchvision](https://github.com/pytorch/vision#installation)（It will be used when running the optimized MNIST dataset, please install the corresponding torchvision according to the version of Pytorch）

More details please see `requirements.txt`.



## Install

1. Clone Repository

   ```bash
   git clone https://github.com/youyve/OptEx.git
   cd OptEx
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Install OptEx

   ```bash
   python setup.py install
   ```



## Test OptEx

```bash
python tests/test_optex_module.py
```



## Run our code

We provide how to use OptEx to optimize three common optimization objective functions: Sphere, Ackley, and Rosenbrock.

```bash
python examples/test_optex.py
```

We also provide an example of using OptEx to optimize the MNIST dataset.

```bash
python examples/minst.py
```

More details please see [`examples/README.md`](examples/README.md)



## BibTeX

```tex
@misc{shu2024optexexpeditingfirstorderoptimization,
      title={OptEx: Expediting First-Order Optimization with Approximately Parallelized Iterations}, 
      author={Yao Shu and Jiongfeng Fang and Ying Tiffany He and Fei Richard Yu},
      year={2024},
      eprint={2402.11427},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.11427}, 
}
```