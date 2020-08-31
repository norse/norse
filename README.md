<p align="center">
<img src="logo.png">
</p>

A library to do [deep learning](https://en.wikipedia.org/wiki/Deep_learning) with [spiking neural networks](https://en.wikipedia.org/wiki/Spiking_neural_network).



<p align="center">
    <a href="https://github.com/norse/norse/actions">
        <img src="https://github.com/norse/norse/workflows/Python%20package/badge.svg" alt="Test status"></a>
    <a href="https://pypi.org/project/norse/" alt="PyPi">
        <img src="https://img.shields.io/pypi/v/norse" />
    </a>
    <a href="https://anaconda.org/norse" alt="Conda">
        <img src="https://img.shields.io/conda/v/norse/norse" />
    </a>
    <a href="https://github.com/norse/norse/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/norse/norse" />
    </a>
    <a href="https://discord.gg/3Xwerqg">
        <img src="https://img.shields.io/discord/723215296399147089"
            alt="chat on Discord"></a>
    <a href="https://www.codacy.com/gh/norse/norse?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=norse/norse&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/a9ab846fc6114afda4320badcb8a69c2"/></a>
    <a href="https://codecov.io/gh/norse/norse">
  <img src="https://codecov.io/gh/norse/norse/branch/master/graph/badge.svg" />
</a>
</p>

This library aims to exploit the advantages of bio-inspired neural components, which are sparse and event-driven - a fundamental difference from artificial neural networks.
Norse expands [PyTorch](https://pytorch.org/) with primitives for bio-inspired neural components, 
bringing you two advantages: a modern and proven infrastructure based on PyTorch and deep learning-compatible spiking neural network components.

**Documentation**: https://norse.ai/docs/

## 1. Getting started

To try Norse, the best option is to run one of the [jupyter notebooks](https://github.com/norse/notebooks/tree/master/) on Google collab. 

Alternatively, [you can install Norse](#installation) and run one of the [included tasks](https://norse.github.io/norse/experiments.html) such as [MNIST](https://en.wikipedia.org/wiki/MNIST_database):
```bash
python -m norse.task.mnist
```


## 2. Using Norse

Norse is generally meant as a library for customized use in specific deep learning tasks. This has been detailed in our documentation: [working with Norse](https://norse.github.io/norse/working.html).

Here we briefly explain how to install Norse and start to apply it in your own work. 

### 2.1. Installation
<a name="installation"></a>

Note that we assume you are using Python version 3.7+, are in a terminal friendly environment, and have installed the necessary requirements. 
[More detailed installation instructions are available in the documentation](https://norse.github.io/norse/installing.html).

<table>
<thead>
<tr>
<th>Method</th><th>Instructions</th><th>Prerequisites</th>
</tr>
</thead>

<tr>
<td>From PyPi</td><td><div class="highlight highlight-source-shell"><pre>
pip install norse
</pre></div></td><td><a href="https://pypi.org/" title="PyPi">Pip</a></td>
</tr>
<tr>
<td>From Conda</td><td> <div class="highlight highlight-source-shell"><pre>
conda install -c norse norse
</pre></div></td><td><a href="https://docs.anaconda.com/anaconda/install/" title="Anaconda">Anaconda</a> or <a href="https://docs.conda.io/en/latest/miniconda.html" title="Miniconda">Miniconda</a></td>
</tr>
<tr>
<td>From source</td><td><div class="highlight highlight-source-shell"><pre>
pip install -qU git+https://github.com/norse/norse
</pre></div></td><td><a href="https://pypi.org/" title="PyPi">Pip</a>, <a href="https://pytorch.org/get-started/locally/" title="PyTorch">PyTorch</a></td>
</tr>
</table>

### 2.2. Running examples

Norse is bundled with a number of example experiments, serving as short, self contained, correct examples ([SSCCE](http://www.sscce.org/)).
They can be run by invoking the `norse` module from the base directory.
More information and tasks are available [in our documentation](https://norse.github.io/norse/experiments.html) and in your console by typing: `python -m norse.task.<task> --help`, where `<task>` is one of the task names.

- To train an MNIST classification network, invoke
    ```bash
    python -m norse.task.mnist
    ```
- To train a CIFAR classification network, invoke
    ```bash
    python -m norse.task.cifar10
    ```
- To train the cartpole balancing task with Policy gradient, invoke
    ```bash
    python -m norse.task.cartpole
    ```

### 2.3. Example on using the library: Long short-term spiking neural networks
The long short-term spiking neural networks from the paper by [G. Bellec, D. Salaj, A. Subramoney, R. Legenstein, and W. Maass (2018)](https://arxiv.org/abs/1803.09574) is one interesting way to apply norse: 
```python
from norse.torch.module import LSNNLayer, LSNNCell
# LSNNCell with 2 input neurons and 10 output neurons
layer = LSNNLayer(LSNNCell, 2, 10)
# Generate data: 20 timesteps with 8 datapoints per batch for 2 neurons
data  = torch.zeros(20, 8, 2)
# Tuple of (output data of shape (8, 2), layer state)
output, new_state = layer.forward(data, state)
```

## 3. Why Norse?

Norse was created for two reasons: to 1) apply findings from decades of research in practical settings
and to 2) accelerate our own research within bio-inspired learning.

A number of projects exist that attempts to leverage the strength of bio-inspired neural networks, 
however only few of them integrate with modern machine-learning libraries such as 
PyTorch or [Tensorflow](https://www.tensorflow.org/) and many of them are no longer actively developed.

We are passionate about Norse and believe it has significant potential outside our own research.
Primarily because we strive to follow best practices and promise to maintain this library for the
simple reason that we depend on it ourselves.
Second, we have implemented a number of neuron models, synapse dynamics, encoding and decoding algorithms, 
dataset integrations, tasks, and examples. While we are far from the comprehensive coverage of 
simulators used in computational neuroscience such as Brian, NEST or NEURON, we expect to close this gap as
we continue to develop the library.

Finally, we are working to keep Norse as performant as possible. 
Preliminary benchmarks suggest that on small networks of up to ~10000 neurons [Norse achieves excellent performance](https://github.com/norse/norse/tree/master/norse/benchmark). We aim to create a library
that scales from a single laptop to several nodes on a HPC cluster. We expect to be significantly
helped in that endeavour by the preexisting investment in scalable training and inference with PyTorch.

[Read more about Norse in our documentation](https://norse.github.io/norse/about.html).

## 4. Similar work
The list of projects below serves to illustrate the state of the art, while explaining our own incentives to create and use norse.

* [BindsNET](https://github.com/BindsNET/bindsnet) also builds on PyTorch and is explicitly targeted at machine learning tasks. It implements a Network abstraction with the typical 'node' and 'connection' notions common in spiking neural network simulators like nest.
* [cuSNN](https://github.com/tudelft/cuSNN) is a C++ GPU-accelerated simulator for large-scale networks. The library focuses on CUDA and includes spike-time dependent plasicity (STDP) learning rules.
* [decolle](https://github.com/nmi-lab/decolle-public) implements an online learning algorithm described in the paper ["Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE)"](https://arxiv.org/abs/1811.10766) by J. Kaiser, M. Mostafa and E. Neftci. 
* [Long short-term memory Spiking Neural Networks (LSNN)](https://github.com/IGITUGraz/LSNN-official) is a tool from the University of Graaz for modelling LSNN cells in [Tensorflow](https://www.tensorflow.org/). The library focuses on a single neuron and gradient model.
* [Nengo](https://www.nengo.ai/nengo-dl/introduction.html) is a neuron simulator, and Nengo-DL is a deep learning network simulator that optimised spike-based neural networks based on an approximation method suggested by [Hunsberger and Eliasmith (2016)](https://arxiv.org/abs/1611.05141). This approach maps to, but does not build on, the deep learning framework Tensorflow, which is fundamentally different from incorporating the spiking constructs into the framework itself. In turn, this requires manual translations into each individual backend, which influences portability.
* [Neuron Simulation Toolkit (NEST)](https://nest-simulator.org) constructs and evaluates highly detailed simulations of spiking neural networks. This is useful in a medical/biological sense but maps poorly to large datasets and deep learning.
* [PyNN](http://neuralensemble.org/docs/PyNN/) is a Python interface that allows you to define and simulate spiking neural network models on different backends (both software simulators and neuromorphic hardware). It does not currently provide mechanisms for optimisation or arbitrary synaptic plasticity.
* [PySNN](https://github.com/BasBuller/PySNN/) is a PyTorch extension similar to Norse. Its approach to model building is slightly different than Norse in that the neurons are stateful.
* [Rockpool](https://gitlab.com/aiCTX/rockpool) is a Python package developed by SynSense for training, simulating and deploying spiking neural networks. It offers both JAX and PyTorch primitives.
* [SlayerPyTorch](https://github.com/bamsumit/slayerPytorch) is a **S**pike **LAY**er **E**rror **R**eassignment library, that focuses on solutions for the temporal credit problem of spiking neurons and a probabilistic approach to backpropagation errors. It includes support for the [Loihi chip](https://en.wikichip.org/wiki/intel/loihi).
* [SNN toolbox](https://snntoolbox.readthedocs.io/en/latest/guide/intro.html) <q>automates the conversion of pre-trained analog to spiking neural networks</q>. The tool is solely for already trained networks and omits the (possibly platform specific) training.
* [SpyTorch](https://github.com/fzenke/spytorch) presents a set of tutorials for training SNNs with the surrogate gradient approach SuperSpike by [F. Zenke, and S. Ganguli (2017)](https://arxiv.org/abs/1705.11146). Norse [implements SuperSpike](https://github.com/norse/norse/blob/master/norse/torch/functional/superspike.py), but allows for other surrogate gradients and training approaches.
* [s2net](https://github.com/romainzimmer/s2net) is based on the implementation presented in [SpyTorch](https://github.com/fzenke/spytorch), but implements convolutional layers as well. It also contains a demonstration how to use those primitives to train a model on the [Google Speech Commands dataset](https://arxiv.org/abs/1804.03209).


## 5. Contributing

Contributions are warmly encouraged and always welcome. However, we also have high expectations around the code base so if you wish to contribute, please refer to our [contribution guidelines](contributing.md).

## 6. Credits

Norse is created by
* [Christian Pehle](https://www.kip.uni-heidelberg.de/people/10110) (@GitHub [cpehle](https://github.com/cpehle/)), doctoral student at University of Heidelberg, Germany.
* [Jens E. Pedersen](https://www.kth.se/profile/jeped) (@GitHub [jegp](https://github.com/jegp/)), doctoral student at KTH Royal Institute of Technology, Sweden.

More information about Norse can be found [in our documentation](https://norse.github.io/norse/about.html).

## 7. License

LGPLv3. See [LICENSE](LICENSE) for license details.
