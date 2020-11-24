.. _page-working:

Working with Norse
-------------------

For us, Norse is a tool to accelerate our own work within spiking neural networks.
This page serves to describe the fundamental ideas behind the Python code in Norse and
provide you with hints as to how you can apply it in your own work
.
We will start by explaining some basic terminology, describe a *suggestion* to how Norse
can be approach, and finally provide examples on how to work with Norse.

Fundamental concepts
=======================

Functional vs. module packaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As with PyTorch, the package is organised into a *functional* and *module* part. 
The *functional* package contains functions and classes that are considered atomic. 
If you use this package, you are expected to piece the code together yourself. 
The *module* package contains PyTorch modules on a slightly higher level.

This is the abstraction we will work with in this document.

How to approach Norse
========================

Norse is meant to be used as a library. Specifically, that means taking parts of it and
remixing to fit the needs of a specific task. 
We have tried to provide useful, documented, and correct features from the spiking neural network domain, such
that they become simple to work with.

The two main differences from artificial neural networks is 1) the state variables containing the neuron parameters
and 2) the temporal dimension (see :ref:`page-spiking`). 
Apart from that, Norse works like you would expect any PyTorch module to work.

A specific recommendation to work with Norse is to consider

1. What neuron models you need to work with
2. What learning/plasticity models you need

Deciding on neuron models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The choice of neuron model depends on the task. 
The `leaky integrate-and-fire neuron model <https://neuronaldynamics.epfl.ch/online/Ch5.S2.html>`_ is one of the
most common. 
In Norse, this is implemented as a recurrent cell in `lif.py <https://github.com/norse/norse/blob/master/norse/torch/module/lif.py#L15>`_.

Deciding on learning/plasiticy models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an area of active development and will be expanded upon soon.

Optimisation is mainly done using PyTorch's optimizers, as seen in the `MNIST task <https://github.com/norse/norse/blob/master/norse/task/mnist.py#L100>`_.

Examples on working with Norse
=================================

We have put considerable effort into streamlining it for three scenarios:
1) porting deep learning models to the spiking/temporal domain,
2) extending existing models, 
3) exploring novel ideas.

Porting deep learning models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A classical example of this can be seen in the `MNIST <https://github.com/norse/norse/blob/master/norse/task/mnist.py>`_
where convolutions are brought into Norse.

Extending existing models
^^^^^^^^^^^^^^^^^^^^^^^^^

An example of this can be seen in the `memory task <https://github.com/norse/norse/blob/master/norse/task/memory.py>`_,
where `adaptive long short-term spiking neural networks <https://github.com/IGITUGraz/LSNN-official>`_ 
are added to 

Exploring novel ideas
^^^^^^^^^^^^^^^^^^^^^

This is an area of active development and will be expanded upon soon.
