.. _page-started:

Getting started
---------------

This page walks you through the initial steps to becoming productive with Norse.
If you would like a more in-depth guide on how to work with Norse, please 
refer to :ref:`page-working`.

If you are entirely unfamiliar with `spiking neural networks (SNNs) <https://en.wikipedia.org/wiki/Spiking_neural_network>`_
we recommend you skim our page that introduces the topic: :ref:`page-spiking`.

Running off-the-shelf code
==========================

If you just want to get started we recommend our `collection of Jupyter Notebook Tutorials <https://github.com/norse/notebooks/>`_.
They can be run online on Google Colab.

Additionally, we provide a set of tasks that you can run right after installing Norse.
One of the most common experiments is the `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_
classification task.
Norse achieve on-par performance with modern non-spiking networks:

.. code:: bash

    python -m norse.task.mnist

Please refer to :ref:`page-tasks` for more tasks and detailed information on how to 
run them.

Building neural networks with state
====================================

If you would like to build your own models with Norse you need to know one thing: **neurons contain state**. 
When you simulate a neuron in Norse you get **two** outputs: the tensor and the state. 
Luckily, Norse initialises all the necessary state *in the beginning*, but you need 
to carry the state onwards.
If you do not, the neuron will never spike and the output will be zero!

.. code:: python

    import torch
    import norse.torch as norse

    cell = norse.LIFFeedForwardCell()
    data = norse.ones(1)
    spikes, state = cell(data)

The *next* time you call the cell, you need to pass in that state. 
Otherwise you will get the exact same output

.. code:: python

    spikes, state = cell(data, state)

*Note*: This is similar to PyTorch's `RNN module <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN>`_ 
if you are looking for inspiration.

Using Norse neurons as PyTorch layers
=====================================

The simplest way to use Norse is to chain neurons as layers in a network,
similar to PyTorch's `Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_.

Unfortunately, this does not work with neurons for the same reason that it does
not work with PyTorch's own RNNs: state.
Luckily, we offer our own `SequentialState <https://norse.github.io/norse/auto_api/norse.torch.module.sequential.html>`_ module:

.. code:: python

    import torch
    import norse.torch as norse

    model = SequentialState(
        torch.nn.Linear(10, 5),
        norse.LIFFeedForwardCell(),
        torch.nn.Linear(5, 1)
    )

    data = torch.ones(8, 10) # (batch, input)

    out, state = model(data)

Using Norse in time
===================

The final step is to include time. Similarly to RNNs/LSTMs in PyTorch we need
to take into account the fact that the neurons state changes. 

This is something we are actively working to simplify.
Currently, the simplest way to go about this is to use the `*Layer` modules
which automatically runs your input in time and then **lift** the regular
PyTorch modules into the time domain (that is, simply run them once for every
timestep):

.. code:: python

    import torch
    import norse.torch as norse

    model = SequentialState(
        norse.Lift(torch.nn.Linear(10, 5)),
        norse.LSNNLayer(5, 5),
        norse.Lift(torch.nn.Linear(5, 1))
    )
    data = torch.ones(100, 8, 10) # (time, batch, input)
    out, state = model(data)

This covers the most basic way to apply Norse. More information can be found
in :ref:`page-spiking`, :ref:`page-working` and :ref:`page-spike-learning`.