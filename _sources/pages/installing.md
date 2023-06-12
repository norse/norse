(page-installing)=

# Installing Norse

We have chosen to build Norse with new Python features such as [type hints](https://docs.python.org/3/whatsnew/3.7.html#whatsnew37-pep560).
For that reason **we require Python version 3.7 or higher**. 
If this is a problem, it is recommended to install Norse via [Docker](https://en.wikipedia.org/wiki/Docker_(software)), as [described below](page-installing-docker).

Norse builds on top of the [PyTorch](https://pytorch.org/) deep learning library, which is also our primary dependency.
This has the benefit that your models are hardware accelerated, providing the [prerequisites are met](https://pytorch.org/get-started/locally/).

## Required dependencies

To install Norse, you need the following two dependencies:
1. `pip` - the [Python package manager](https://pypi.org/project/pip/)
   * This is preinstalled in most Linux and Unix distros
   * Note that Norse requires Python >= 3.8
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
# Or, using CUDA
docker pull quay.io/norse/norse:latest-cuda
```

### Installing from source
```bash
git clone https://github.com/norse/norse
cd norse
python setup.py install
```


## Optional dependencies

Some of the tasks require additional dependencies like [Pytorch Lightning](https://pytorchlightning.ai/), [Torchtext](https://pytorch.org/text/stable/index.html) and [Torchvision](https://pytorch.org/docs/stable/torchvision/index.html).
We also offer support for [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) to make it easier to visualise the training and introspect models.

(page-installing-docker)=
## Running Norse notebooks with Docker

Docker creates a closed environment for you, which also means that the network and
filesystem is isolated. Without going into details, here are three steps you can
take to create a [Jupyter Notebook](https://jupyter.org/) environment with
Docker. You will have to replace `/your/directory` with the **full** path to
your current directory.

```bash
docker run -it -p 8888:8888 -v /your/directory:/work quay.io/norse/norse bash
pip3 install jupyter
jupyter notebook --notebook-dir=/work --ip 0.0.0.0 --allow-root
```

The command line will now show you a URL you can copy-paste into your browser.
And voila!

### GPU acceleration in Docker

If you would like to have GPU hardware acceleration when running the `latest-cuda` version of the
docker container, you will have to enable the NVIDIA runtime, 
as described here: https://developer.nvidia.com/nvidia-container-runtime.

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

This is likely because an existing, incompatible version of PyTorch is interferring with the installation.
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