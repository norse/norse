.. _page-working:

Working with Norse
-------------------

For us, Norse is a tool to accelerate our own work within spiking neural networks.
This page serves to describe the fundamental ideas behind the Python code in Norse and
provide you with specific tools to become productive with SNNs.

We will start by explaining some basic terminology, describe a *suggestion* to how Norse
can be approached, and finally provide examples on how we have solved specific problems
with Norse.

Fundamental concepts
=======================

Functional vs. module packaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As with PyTorch, the package is organised into a *functional* and *module* part. 
The *functional* package contains functions and classes that are considered atomic. 
If you use this package, you are expected to piece the code together yourself. 
The *module* package contains PyTorch modules on a slightly higher level.

We will work with the *module* abstraction in this document.

Neuron state
^^^^^^^^^^^^

Neurons have parameters that determine their function. For example, they have a
certain membrane voltage that will lead the neuron to spike *if* the voltage is
above a threshold. Someone needs to keep track of that membrane voltage. If we 
wouldn't, the neuron membrane would never update and we would never get any
spikes. In Norse, we refer to that as the **neuron state**.

In code, it looks like this:

.. code:: python

    import torch
    import norse.torch as norse

    cell = norse.LIFCell()
    data = norse.ones(1)
    spikes, state = cell(data)        # First run is done without any state
    ...
    spikes, state = cell(data, state) # Now we pass in the previous state

Neuron dynamics and time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Norse solves two of the hardest parts about running neuron simulations: neural equations and temporal dynamics.
We have solved that by providing four modules for each neuron type.
As an example, the `Long short-term memory neuron model <https://arxiv.org/abs/1803.09574>`_ the four modules are:

* ``LSNNCell``
* ``LSNNRecurrentCell``
* ``LSNN`` and
* ``LSNNRecurrent``

The four modules represent the combinations between simulating over time and having recurrence.
In other words, the ``LSNNCell`` is *not* recurrent, and expects the input data to *not* have time, while the
``LSNNRecurrent`` *is* recurrent and expects the input to have time in the *first* dimension.

How to approach Norse
=====================

Norse is meant to be used as a library. Specifically, that means taking parts of it and
remixing to fit the needs of a specific task. 
We have tried to provide useful, documented, and correct features from the spiking neural network domain, such
that they become simple to work with.

The two main differences from artificial neural networks is 1) the state variables containing the neuron parameters
and 2) the temporal dimension (see :ref:`page-spiking`). 
Apart from that, Norse works like you would expect any PyTorch module to work.

When working with Norse we recommend that you consider two things

1. Neuron models 
2. Learning algorithms and/or plasticity models 

Deciding on neuron models
^^^^^^^^^^^^^^^^^^^^^^^^^

The choice of neuron model depends on the task. 
The `leaky integrate-and-fire neuron model <https://neuronaldynamics.epfl.ch/online/Ch5.S2.html>`_ is one of the
most common. 
In Norse, this is implemented as a recurrent cell in `lif.py <https://github.com/norse/norse/blob/master/norse/torch/module/lif.py#L15>`_.

Deciding on learning/plasiticy models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimisation is mainly done using PyTorch's optimizers, as seen in the `MNIST task <https://github.com/norse/norse/blob/master/norse/task/mnist.py#L100>`_.

Examples on working with Norse
=================================

Porting deep learning models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A classical example of this can be seen in the `MNIST <https://github.com/norse/norse/blob/master/norse/task/mnist.py>`_
where convolutions are brought into Norse.

Extending existing models
^^^^^^^^^^^^^^^^^^^^^^^^^

An example of this can be seen in the `memory task <https://github.com/norse/norse/blob/master/norse/task/memory.py>`_,
where `adaptive long short-term spiking neural networks <https://github.com/IGITUGraz/LSNN-official>`_ 
are added to 