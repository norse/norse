.. _page-working:

Working with Norse
-------------------

For us, Norse is a tool to accelerate our own work within spiking neural networks.
This page serves to describe the fundamental ideas behind the Python code in Norse.
We will start by explaining some basic terminology, describe a *suggestion* to how Norse
can be approach, and finally provide examples on how to work with Norse.


Fundamental concepts
=======================

Functional vs. module packaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As with PyTorch, the package is organised into a *functional* and *module* part. 
The *functional* package contains functions and classes that are considered atomic. If you use this package, you are expected to piece the code together yourself. The *module* package contains PyTorch modules on a slightly higher level. This is the abstraction we will work with in this document.

Cell parameters
^^^^^^^^^^^^^^^^^^^

All cells and layers use neuron parameters to describe information about the neurons (for instance the membrane potential and leak).
Norse is designed such that all neurons require parameters for each forward pass.
In practice this looks like

.. code:: python

    import torch
    from norse.torch.module.lif import LIFParameters, LIFCell
    data = torch.ones(2)
    p    = LIFParameters(v_leak=1.0) # Default values except for v_leak
    cell = LIFCell(2, 4, p=p)     # Shape 2 -> 4

All future executions with that cell will then use the neuron parameters.

Layer and cell states
^^^^^^^^^^^^^^^^^^^^^^^^^
Because we are working with temporal data, neuron parameters change over time. 
That is encoded in a state class (e.g. `LIFState <https://github.com/norse/norse/blob/012a97bb23ea6b6ec0cb47866c62b3711b0c53da/norse/torch/functional/lif.py#L39>`_) 
that represents the current values of the neuron (for instance membrane potential and leak).

Neurons and states **are immutable**. 
Every time you call a neuron, the state will be returned along with the output (spikes). 
Typically as the second parameter.
The reason for this is that you can then re-use the state in the next time step, to capture the neuron parameters.

The initial value of such a state is found by invoking the ``initial_state`` of the cell/layer, like so:

.. code:: python

    import torch
    from norse.torch.module.lif import LIFCell
    data = torch.ones(2)
    cell = LIFCell(2, 4)     # Shape 2 -> 4
    state = cell.initial_state(8, "cpu")

This informs the cell that there are 8 entries in the batches and that the data will be evaluated on the CPU.

Here is an example on how to evaluate a sequence of inputs modified from the `LIFLayer <https://github.com/norse/norse/blob/master/norse/torch/module/lif.py#L106>`_.


.. code:: python

    import torch
    from norse.torch.module.lif import LIFCell
    inputs = torch.ones(100, 8, 2) # Shape (timesteps, batch, input)
    
    cell = LIFCell(2, 4)           # Shape (input, output)
    state = cell.initial_state(8, "cpu")
    outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

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
