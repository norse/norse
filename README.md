<p align="center">
<img src="https://raw.githubusercontent.com/norse/norse/main/logo.png">
</p>

A [deep learning](https://en.wikipedia.org/wiki/Deep_learning) library for [spiking neural networks](https://en.wikipedia.org/wiki/Spiking_neural_network).

<p align="center">
    <a href="https://github.com/norse/norse/actions">
        <img src="https://github.com/norse/norse/workflows/Build%20Python/badge.svg" alt="Test status"></a>
    <a href="https://pypi.org/project/norse/" alt="PyPi Norse">
        <img src="https://img.shields.io/pepy/dt/norse?label=PyPi%20Downloads" />
    </a>
    <a href="https://anaconda.org/conda-forge/norse" alt="Conda Norse repository">
        <img src="https://img.shields.io/conda/dn/conda-forge/norse?label=Conda%20Downloads" />
    </a>
    <a href="https://github.com/norse/norse/pulse" alt="Activity">
        <img src="https://img.shields.io/github/last-commit/norse/norse" />
    </a>
    <a href="https://discord.gg/bKsG4tN9wT">
        <img src="https://img.shields.io/discord/723215296399147089"
            alt="chat on Discord"></a>
    <a href="https://www.codacy.com/gh/norse/norse?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=norse/norse&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/a9ab846fc6114afda4320badcb8a69c2"/></a>
    <a href="https://codecov.io/gh/norse/norse"><img src="https://codecov.io/gh/norse/norse/branch/master/graph/badge.svg" /></a>
    <a href="https://doi.org/10.5281/zenodo.4422025"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4422025.svg" alt="DOI"></a>
</p>

Norse aims to exploit the advantages of bio-inspired neural components, which are sparse and event-driven - a fundamental difference from artificial neural networks.
Norse expands [PyTorch](https://pytorch.org/) with primitives for bio-inspired neural components, 
bringing you two advantages: a modern and proven infrastructure based on PyTorch and deep learning-compatible spiking neural network components.

**Documentation**: [norse.github.io/norse/](https://norse.github.io/norse/)

## 1. Getting started

The fastest way to try Norse is via the [jupyter notebooks on Google collab](https://github.com/norse/notebooks/tree/main/).

## 2. Using Norse

Norse presents plug-and-play components for deep learning with spiking neural networks.
Here, we describe how to install Norse and start to apply it in your own work.
[Read more in our documentation](https://norse.github.io/norse/working.html).

### 2.1. Installation
<a name="installation"></a>

We assume you are using **Python version 3.8+** and have **installed PyTorch version 1.9 or higher**. 
[Read more about the prerequisites in our documentation](https://norse.github.io/norse/installing.html).

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
<td>From source</td><td><div class="highlight highlight-source-shell"><pre>
pip install -qU git+https://github.com/norse/norse
</pre></div></td><td><a href="https://pypi.org/" title="PyPi">Pip</a>, <a href="https://pytorch.org/get-started/locally/" title="PyTorch">PyTorch</a></td>
</tr>
<tr>
<td>With Docker</td><td><div class="highlight highlight-source-shell"><pre>
docker pull quay.io/norse/norse
</pre></div></td><td><a href="https://www.docker.com/get-started" title="Docker">Docker</a></td>
</tr>
<tr>
<td>From Conda</td><td> <div class="highlight highlight-source-shell"><pre>
conda install norse
</pre></div></td><td><a href="https://docs.anaconda.com/anaconda/install/" title="Anaconda">Anaconda</a> or <a href="https://docs.conda.io/en/latest/miniconda.html" title="Miniconda">Miniconda</a></td>
</tr>
</table>

For troubleshooting, please refer to our [installation guide](https://norse.github.io/norse/pages/installing.html#installation-troubleshooting), [create an issue on GitHub](https://github.com/norse/norse/issues) or [write us on Discord](https://discord.gg/7fGN359).

### 2.2. Running examples

Norse is bundled with a number of example tasks, serving as short, self contained, correct examples ([SSCCE](http://www.sscce.org/)).
They can be run by invoking the `norse` module from the base directory.
More information and tasks are available [in our documentation](https://norse.github.io/norse/tasks.html) and in your console by typing: `python -m norse.task.<task> --help`, where `<task>` is one of the task names.

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

Norse is compatible with [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/),
as demonstrated in the [PyTorch Lightning MNIST task variant](https://github.com/norse/norse/blob/main/norse/task/mnist_pl.py) (requires PyTorch lightning):

```bash
python -m norse.task.mnist_pl --gpus=4
```

### 2.3. Example: Spiking convolutional classifier 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/norse/notebooks/blob/main/mnist_classifiers.ipynb)

This classifier is taken from our [tutorial on training a spiking MNIST classifier](https://github.com/norse/notebooks#level-intermediate) and achieves >99% accuracy.

```python
import torch, torch.nn as nn
from norse.torch import LICell             # Leaky integrator
from norse.torch import LIFCell            # Leaky integrate-and-fire
from norse.torch import SequentialState    # Stateful sequential layers

model = SequentialState(
    nn.Conv2d(1, 20, 5, 1),      # Convolve from 1 -> 20 channels
    LIFCell(),                   # Spiking activation layer
    nn.MaxPool2d(2, 2),
    nn.Conv2d(20, 50, 5, 1),     # Convolve from 20 -> 50 channels
    LIFCell(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),                # Flatten to 800 units
    nn.Linear(800, 10),
    LICell(),                    # Non-spiking integrator layer
)

data = torch.randn(8, 1, 28, 28) # 8 batches, 1 channel, 28x28 pixels
output, state = model(data)      # Provides a tuple (tensor (8, 10), neuron state)
```

### 2.4. Example: Long short-term spiking neural networks
The long short-term spiking neural networks from the paper by [G. Bellec, D. Salaj, A. Subramoney, R. Legenstein, and W. Maass (2018)](https://arxiv.org/abs/1803.09574) is another interesting way to apply norse: 
```python
import torch
from norse.torch import LSNNRecurrent
# Recurrent LSNN network with 2 input neurons and 10 output neurons
layer = LSNNRecurrent(2, 10)
# Generate data: 20 timesteps with 8 datapoints per batch for 2 neurons
data  = torch.zeros(20, 8, 2)
# Tuple of (output spikes of shape (20, 8, 2), layer state)
output, new_state = layer(data)
```

## 3. Why Norse?

Norse was created for two reasons: to 1) apply findings from decades of research in practical settings
and to 2) accelerate our own research within bio-inspired learning.

We are passionate about Norse: we strive to follow best practices and promise to maintain this library for the
simple reason that we depend on it ourselves.
We have implemented a number of neuron models, synapse dynamics, encoding and decoding algorithms, 
dataset integrations, tasks, and examples.
Combined with the PyTorch infrastructure and our high coding standards, we have found Norse to be an excellent tool for modelling *scaleable* experiments and [Norse is actively being used in research](https://norse.github.io/norse/papers.html).

Finally, we are working to keep Norse as performant as possible. 
Preliminary benchmarks suggest that Norse [achieves excellent performance on small networks of up to ~5000 neurons per layer](https://github.com/norse/norse/tree/main/norse/benchmark). 
Aided by the preexisting investment in scalable training and inference with PyTorch, Norse scales from a single laptop to several nodes on an HPC cluster with little effort.
As illustrated by our [PyTorch Lightning example task](https://norse.github.io/norse/tasks.html#mnist-in-pytorch-lightning).


[Read more about Norse in our documentation](https://norse.github.io/norse/about.html).

## 4. Similar work

We refer to the [Neuromorphic Software Guide](https://open-neuromorphic.org/neuromorphic-computing/software/) for a comprehensive list of software for neuromorphic computing.


## 5. Contributing

Contributions are warmly encouraged and always welcome. However, we also have high expectations around the code base so if you wish to contribute, please refer to our [contribution guidelines](contributing.md).

## 6. Credits

Norse is created by
* [Christian Pehle](https://www.kip.uni-heidelberg.de/people/10110) (@GitHub [cpehle](https://github.com/cpehle/)), PostDoc at University of Heidelberg, Germany.
* [Jens E. Pedersen](https://www.kth.se/profile/jeped) (@GitHub [jegp](https://github.com/jegp/)), doctoral student at KTH Royal Institute of Technology, Sweden.

More information about Norse can be found [in our documentation](https://norse.github.io/norse/about.html). The research has received funding from the EC Horizon 2020 Framework Programme under Grant Agreements 785907 and 945539 (HBP) and by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy EXC 2181/1 - 390900948 (the Heidelberg STRUCTURES Excellence Cluster).

## 7. Citation

If you use Norse in your work, please cite it as follows:

```BibTex
@software{norse2021,
  author       = {Pehle, Christian and
                  Pedersen, Jens Egholm},
  title        = {{Norse -  A deep learning library for spiking 
                   neural networks}},
  month        = jan,
  year         = 2021,
  note         = {Documentation: https://norse.ai/docs/},
  publisher    = {Zenodo},
  version      = {0.0.7},
  doi          = {10.5281/zenodo.4422025},
  url          = {https://doi.org/10.5281/zenodo.4422025}
}
```

Norse is actively applied and cited in the literature. We refer to [Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=de&user=A4VwxccAAAAJ&citation_for_view=A4VwxccAAAAJ:2osOgNQ5qMEC) or [Semantic Scholar](https://www.semanticscholar.org/paper/Norse-A-deep-learning-library-for-spiking-neural-Pehle-Pedersen/bdd21dfe8c4a503365a49bfdb099e63c74823c7c) for a list of citations.

## 8. License

LGPLv3. See [LICENSE](LICENSE) for license details.
