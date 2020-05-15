<p align="center">
<img src="logo.png">
</p>

A library to do [deep learning](https://en.wikipedia.org/wiki/Deep_learning) with [spiking neural networks](https://en.wikipedia.org/wiki/Spiking_neural_network).


[![Test Status](https://github.com/norse/norse/workflows/Python%20package/badge.svg)](https://github.com/norse/norse/actions) 

The purpose of this library is to exploit the advantages of bio-inspired neural components, which are sparse and event-driven - a fundamental difference from artificial neural networks.
Norse expands [PyTorch](https://pytorch.org/) with primitives for bio-inspired neural components, 
bringing you two advantages: a modern and proven infrastructure based on PyTorch and deep learning-compatible spiking neural network components.

**Documentation**: https://norse.ai/docs/

## 1. Example usage: template tasks

Norse comes packed with a few example tasks, such as [MNIST](https://en.wikipedia.org/wiki/MNIST_database), but is generally meant for use in specific deep learning tasks (see below section on long short-term spiking neural networks):
```bash
python -m norse mnist
```
You can also run one of the [jupyter notebooks](notebooks/) on google collab.


## 2. Getting Started

Norse is a machine learning library that builds on the [PyTorch](https://pytorch.org/) infrastructure. 
While we have a few tasks included, it is meant to be used in designing and evaluating experiments involving biologically realistic neural networks.

This readme explains how to install norse and apply it in your own experiments. If you just want to try out the library, perhaps the best option is to run one of the [jupyter notebooks](notebooks/) on google collab. 

### 2.1. Installation

Note that this guide assumes you are on a terminal friendly environment with access to the `pip`, `python` and `git` commands. Python version 3.7+ is required.

#### 2.1.1. Installing from source

For now this is the recommended way of installing the package, make sure
that you have installed torch, following their [installation instructions](https://pytorch.org/get-started/locally/)
and then install norse.

You can either directly install the library from github using pip:

```bash
git clone https://github.com/norse/norse
cd norse
pip install -e .
```

#### 2.1.2. Installing from PyPi

```bash
pip install norse
```


The primary dependencies of this project are [torch](https://pytorch.org/), [tensorboard](https://www.tensorflow.org/tensorboard/) and [OpenAI gym](https://github.com/openai/gym).
A more comprehensive list of dependencies can be found in [`requirements.txt`](requirements.txt).

### 2.2. Running examples

The directory [norse/task](norse/task) contains three example experiments, serving as short, self contained, correct examples ([SSCCE](http://www.sscce.org/)).
You can execute them by invoking the `norse` module from the base directory.

- To train an MNIST classification network, invoke
    ```
    python -m norse mnist
    ```
- To train a CIFAR classification network, invoke
    ```
    python -m norse cifar
    ```
- To train the cartpole balancing task with Policy gradient, invoke
    ```
    python -m norse gym
    ```
    
The default choices of hyperparameters are meant as reasonable starting points. More information is available when typing: `python -m norse --help`.

### 2.3. Example on using the library: Long short-term spiking neural networks
The long short-term spiking neural networks from the paper by [G. Bellec, D. Salaj, A. Subramoney, R. Legenstein, and W. Maass (2018)](https://arxiv.org/abs/1803.09574) is one interesting way to apply norse: 
```python
from norse.torch.module import LSNNLayer, LSNNCell
# LSNNCell with 2 inputs and 10 outputs
layer = LSNNLayer(LSNNCell, 2, 10)
# 5 batch size running on CPU
state = layer.initial_state(5, "cpu") 
# Generate data of shape [5, 2, 10]
data  = torch.zeros(2, 5, 2)
# Tuple of output data and layer state
output, new_state = layer.forward(data, state) 
```

## 3. Similar work

A number of projects exist that attempts to leverage the strength of bio-inspired neural networks, however none of them are fully integrated with modern machine-learning libraries such as Torch or [Tensorflow](https://www.tensorflow.org/). 
Norse was created for two reasons: to 1) apply findings from decades of research in practical settings, and to 2) accelerate our own research within bio-inspired learning.

The below list of projects serves to illustrate the state of the art, while explaining our own incentives to create and use norse.

* [BindsNET](https://github.com/BindsNET/bindsnet) also builds on PyTorch and is explicitly targeted at machine learning tasks. It implements a Network abstraction with the typical 'node' and 'connection' notions common in spiking neural network simulators like nest.
* [Long short-term memory Spiking Neural Networks (LSNN)](https://github.com/IGITUGraz/LSNN-official) is a tool from the University of Graaz for modelling LSNN cells in [Tensorflow](https://www.tensorflow.org/). The library focuses on a single neuron and gradient model.
* [Nengo](https://www.nengo.ai/nengo-dl/introduction.html) is a neuron simulator, and Nengo-DL is a deep learning network simulator that optimised spike-based neural networks based on an approximation method suggested by [Hunsberger and Eliasmith (2016)](https://arxiv.org/abs/1611.05141). This approach maps to, but does not build on, the deep learning framework Tensorflow, which is fundamentally different from incorporating the spiking constructs into the framework itself. In turn, this requires manual translations into each individual backend, which influences portability.
* [Neuron Simulation Toolkit (NEST)](https://nest-simulator.org) constructs and evaluates highly detailed simulations of spiking neural networks. This is useful in a medical/biological sense but maps poorly to large datasets and deep learning.
* [PyNN](http://neuralensemble.org/docs/PyNN/) is a Python interface that allows you to define and simulate spiking neural network models on different backends (both software simulators and neuromorphic hardware). It does not currently provide mechanisms for optimisation or arbitrary synaptic plasticity.
* [SlayerPyTorch](https://github.com/bamsumit/slayerPytorch) is a **S**pike **LAY**er **E**rror **R**eassignment library, that focuses on solutions for the temporal credit problem of spiking neurons and a probabilistic approach to backpropagation errors. It includes support for the [Loihi chip](https://en.wikichip.org/wiki/intel/loihi).
* [SNN toolbox](https://snntoolbox.readthedocs.io/en/latest/guide/intro.html) <q>automates the conversion of pre-trained analog to spiking neural networks</q>. The tool is solely for already trained networks and omits the (possibly platform specific) training.
* [SpyTorch](https://github.com/fzenke/spytorch) presents a set of tutorials for training SNNs with the surrogate gradient approach SuperSpike by [F. Zenke, and S. Ganguli (2017)](https://arxiv.org/abs/1705.11146). Norse [implements SuperSpike](https://github.com/norse/norse/blob/master/norse/torch/functional/superspike.py), but allows for other surrogate gradients and training approaches.
* [s2net](https://github.com/romainzimmer/s2net) is based on the implementation presented in [SpyTorch](https://github.com/fzenke/spytorch), but implements convolutional layers as well. It also contains a demonstration how to use those primitives to train a model on the [Google Speech Commands dataset](https://arxiv.org/abs/1804.03209).

## 4. Contributing

Please refer to the [contributing.md](contributing.md)

## 5. Credits

Norse is created by
* [Christian Pehle](https://www.kip.uni-heidelberg.de/people/10110) (@GitHub [cpehle](https://github.com/cpehle/)), doctoral student at University of Heidelberg, Germany.
* [Jens E. Pedersen](https://www.kth.se/profile/jeped) (@GitHub [jegp](https://github.com/jegp/)), doctoral student at KTH Royal Institute of Technology, Sweden.


## 6. License

LGPLv3. See [LICENSE](LICENSE) for license details.
