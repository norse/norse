(page-installing)=

# Installing Norse

Norse **requires Python version 3.9 or higher**. 
If this is a problem, it is recommended to install Norse via [Docker](https://en.wikipedia.org/wiki/Docker_(software)), as [described below](page-installing-docker).

Norse builds on top of the [PyTorch](https://pytorch.org/) deep learning library, which is also our primary dependency.
This has the benefit that your models are hardware accelerated, providing the [prerequisites are met](https://pytorch.org/get-started/locally/).

## Required dependencies

To install Norse, you need the following two dependencies:
1. `pip` - the [Python package manager](https://pypi.org/project/pip/)
   * This is preinstalled in most Linux and Unix distros
   * Note that Norse requires Python >= 3.9
2. `torch` - the [deep learning accelerator](https://pytorch.org/get-started/locally/)
   * Please follow the guide available here https://pytorch.org/get-started/locally/
   * Select the CUDA version if you require GPU hardware acceleration
  
## Installing Norse

Note that the following commands require access to a 
[command line interface](https://en.wikipedia.org/wiki/Command-line_interface).

### Installing with [`pip`](https://pypi.org/project/pip/)
```bash
pip install norse
```

### Installing from [Conda](https://docs.conda.io/en/latest/)
```bash
conda install -c norse norse
```

### Installing with [Docker](https://en.wikipedia.org/wiki/Docker_(software))
```bash
docker pull quay.io/norse/norse
```

Alternatively, build from source:

```bash
docker build -t norse -f publish/Dockerfile --build-arg VERSION=1.0.0 .
```

Replace `1.0.0` with the desired version number.

### Installing from source
```bash
git clone https://github.com/norse/norse
cd norse
pip install -e .
```


## Optional dependencies

Some of the tasks require additional dependencies like [Pytorch Lightning](https://pytorchlightning.ai/), [Torchtext](https://pytorch.org/text/stable/index.html) and [Torchvision](https://pytorch.org/docs/stable/torchvision/index.html).
We also offer support for [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) to make it easier to visualise the training and introspect models.

(page-installing-docker)=
## Running Norse with Docker

Docker creates a closed environment for you, which also means that the network and
filesystem is isolated. To run with a mounted volume:

```bash
docker run -it --rm -v $(pwd):/workspace quay.io/norse/norse
```

### GPU acceleration in Docker

The Docker image includes CUDA support by default. To use GPU hardware acceleration, you need:

1. An NVIDIA GPU
2. The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
3. The `--gpus all` flag when running the container

```bash
docker run -it --rm --gpus all quay.io/norse/norse python -c "import torch; print(torch.cuda.is_available())"
```

The `--ipc=host` flag is recommended for multiprocessing and DataLoader with multiple workers:

```bash
docker run -it --rm --gpus all --ipc=host -v $(pwd):/workspace quay.io/norse/norse
```

For more information on hardware acceleration, please refer to our page on {ref}(page-hardware).


## Installation troubleshooting

Below follows a list of known problems that hopefully address your problem. 
If not, please do not hesitate to reach out either by
* Creating an issue on GitHub: https://github.com/norse/norse/issues
* Registering on our [Discord] chat server: https://discord.gg/7fGN359

Because we are relying on optimised C++ for some of the hotspots in the library, you will need to download and install  [CMake](https://cmake.org/) and [PyTorch](https://pytorch.org/get-started/locally/) *before* you can install Norse.
For that reason, we recommend [following the PyTorch "Get Started" guide](https://pytorch.org/get-started/locally/) as the first step.

You might also have to install Python headers if you have not already done that.
In Debian-based distros (like Ubuntu), this can be done by running `apt install python3-dev`.

### Common problems

```{admonition} ImportError: ... /norse_op.so: undefined symbol: _ZN2at5addmmERKNS_6TensorES2_S2_RKN3c106ScalarES6_
:class: warning

This is likely because an existing, incompatible version of PyTorch is interfering with the installation.
Try to 
1. remove PyTorch (`pip uninstall torch`),
2. install `torch` by following [the official guide at pytorch.org](https://pytorch.org/get-started/locally/), and
3. reinstall Norse with your preferred method

See this issue for more information: https://github.com/norse/norse/issues/280
```

```{admonition} UnsatisfiableError: The following specifications were found to be incompatible with each other
:class: warning

This can happen during installation from Conda of Norse<=0.07.
A solution is to add the `conda-forge` channel, like so: `conda install -c norse -c conda-forge norse`

Or, when creating an environment: `conda create -d -c norse -n temptest2 python==3.9 norse==0.0.7`
```
