<p align="center">
<img src="https://raw.githubusercontent.com/norse/norse/master/logo.png">
</p>

A [deep learning](https://en.wikipedia.org/wiki/Deep_learning) library for [spiking neural networks](https://en.wikipedia.org/wiki/Spiking_neural_network).

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
        <img src="https://img.shields.io/github/last-commit/norse/norse" />
    </a>
    <a href="https://discord.gg/7fGN359">
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

## Getting started

To try Norse, the best option is to run one of the [jupyter notebooks](https://github.com/norse/notebooks/tree/master/) on Google collab. 

Alternatively, [you can install Norse](#installation) and run one of the [included tasks](https://norse.github.io/norse/tasks.html) such as [MNIST](https://en.wikipedia.org/wiki/MNIST_database):
```bash
python -m norse.task.mnist
```

The {ref}`page-started` and {ref}`page-working` pages show how to build your own models with Norse while explaining a few fundamental concepts around spiking neural networks.

## Installing Norse

Note that we assume you are using Python version 3.7+, are in a terminal friendly environment, and have installed the necessary requirements, 
depending on your installation method. 
More detailed installation instructions are available here: {ref}`page-installing`.

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
conda install -c norse norse
</pre></div></td><td><a href="https://docs.anaconda.com/anaconda/install/" title="Anaconda">Anaconda</a> or <a href="https://docs.conda.io/en/latest/miniconda.html" title="Miniconda">Miniconda</a></td>
</tr>
</table>


### Running examples

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
as demonstrated in the [PyTorch Lightning MNIST task variant](https://github.com/norse/norse/blob/master/norse/task/mnist_pl.py) (requires PyTorch lightning):

```bash
python -m norse.task.mnist_pl --gpus=4
```

Read more in our {ref}`page-spiking` and visit our [Jupyter Notebook examples](https://github.com/norse/notebooks). 

## Advanced uses and opimizations

Norse is meant to be used as a library for spiking neural networks in customized deep learning models.
This typically means porting other models to the spiking/temporal domain, 
extending existing models, 
or starting completely from scratch. 
All three use cases are motivated and briefly described in {ref}`page-working`.

Unfortunately, spiking neural networks are resource intensive.
The page on {ref}`page-hardware` explains how to accelerate the simulations using dedicated hardware.

## Contributing

Contributions are warmly encouraged and always welcome. However, we also have high expectations around the code base so if you wish to contribute, please refer to our [contribution guidelines](https://github.com/norse/norse/blob/master/contributing.md).

## Credits

Norse is created by
* [Christian Pehle](https://www.kip.uni-heidelberg.de/people/10110) (@GitHub [cpehle](https://github.com/cpehle/)), PostDoc at University of Heidelberg, Germany.
* [Jens E. Pedersen](https://www.kth.se/profile/jeped) (@GitHub [jegp](https://github.com/jegp/)), doctoral student at KTH Royal Institute of Technology, Sweden.

More information about Norse can be found [in our documentation](https://norse.github.io/norse/about.html). The research has received funding from the EC Horizon 2020 Framework Programme under Grant Agreements 785907 and 945539 (HBP) and by the Deutsche Forschungsgemeinschaft (DFG, German Research Fundation) under Germany's Excellence Strategy EXC 2181/1 - 390900948 (the Heidelberg STRUCTURES Excellence Cluster).

## Citation

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
  version      = {0.0.6},
  doi          = {10.5281/zenodo.4422025},
  url          = {https://doi.org/10.5281/zenodo.4422025}
}
```

Norse is actively applied and cited in the literature. We are keeping track of the papers cited by Norse [in our documentation](https://norse.github.io/norse/papers.html).

## License

LGPLv3. See [LICENSE](LICENSE) for license details.
