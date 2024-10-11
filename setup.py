from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="OptEx",
    version="0.1.0",
    author="Yao Shu, Lianzhong You",
    author_email="shuyao@gml.ac.cn, youlianzhong@gml.ac.cn",
    description="A first-order optimization acceleration framework based on PyTorch and GPyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/youyve/OptEx",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "torch>=1.10",
        "gpytorch>=1.10",
        "matplotlib",
    ],
)
