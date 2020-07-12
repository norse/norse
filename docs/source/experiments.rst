.. _page-experiments:

Running Experiments
-------------------

Norse arrives with a number of built-in examples. 
These tasks serves to 1) illustrate what types of tasks *can* be done with Norse
and 2) how to use Norse for specific experiments. Please note that some tasks require
additional dependencies, like OpenAI gym for the cartpole task, which is not included in 
vanilla Norse.

Parameters
==========

The tasks below uses a large number of configurable parameters to control the model/network size, 
`epochs and batch size <https://pytorch.org/docs/stable/tensor_attributes.html#torch-device>`_,
task load, `pytorch device <https://pytorch.org/docs/stable/tensor_attributes.html#torch-device>`_, 
`learning rate <https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10>`_,
and many more parameters. 

Another very important parameter determines which backpropagation model to use.
This is particularly important for spiking neural network models and is described in more 
detail on the page about :ref:`page-spike-learning`.

All programs below are built with `Abseil Python <https://github.com/abseil/abseil-py>`_ which
gives you extensive command line interface (CLI) help descriptions, where you can look up further
descriptions - and find even more - of the parameters above. You can access these by
using the ``--help`` flag on any tasks below, for instance ``python -m norse.task.mnist --help``.

Cartpole
========

This tasks is a balancing exercise where a controller learns to counter the gravitational force
on an upright cartpole. You will need to install `OpenAI Gym <https://gym.openai.com/>`_, to
provide the simulation environments for the robot. If you are using pip, this is as easy as
typing ``pip install gym``.

The cartpole task can be run like so:

.. code:: bash

    python3 -m norse.task.cartpole

Cifar10
=======

Cifar10 is a labeled database of 60'000 32x32 images in 10 classes. The task is to learn to classify
each image.

The cifar task can be run without any additional dependencies like so:

.. code:: bash

    python3 -m norse.task.cifar10


Correlation experiment
======================

The correlation experiment serves to demonstrate how neurons can learn patterns with a certain probability. 
It can be run without any additional dependencies like so:

.. code:: bash

    python3 -m norse.task.correlation_experiment

Memory task
===========

The memory task demonstrates how a recurrent spiking neural network can store a pattern
and later recall it, similar to the STORE/RECALL experiment in the paper on
`Biologically inspired alternatives to backpropagation through time for learning in recurrent neural nets <https://arxiv.org/abs/1901.09049>`_ by Guillaume Bellec, Franz Scherr, Elias Hajek, 
Darjan Salaj, Robert Legenstein, and Wolfgang Maass.

The task can be run without any additional dependencies like so:

.. code:: bash

    python3 -m norse.task.memory

MNIST
=====

MNIST is a `database of 70'000 handwritten digits <https://en.wikipedia.org/wiki/MNIST_database>`_ where
the task is to learn to classify each image as one of the 10 digits.

The task can be run without any additional dependencies like so:

.. code:: bash

    python3 -m norse.task.mnist