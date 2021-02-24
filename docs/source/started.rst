.. _page-started:

Quickstart
----------

This page walks you through the initial steps to becoming productive with Norse.
We will cover how to 

* Work with neuron state
* Work with Norse without time
* Work with Norse with recurrence
* Work with Norse with time

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

If you would like to build your own models with Norse you need to know that **neurons contain state**. 
In practice, that meant that **neurons in Norse outputs two things**: a spike tensor and the neuron state. 
Norse initialises all the necessary state for you *in the beginning*, but you need 
to carry the state onwards.
If you do not, the state will always be zero, the neuron will never spike and your neurons will be 
forever dead!

.. code:: python

    import torch
    import norse

    cell = norse.torch.LIFCell()
    data = torch.ones(1)
    spikes, state = cell(data)

The *next* time you call the cell, you need to pass in that state. 
Otherwise you will get the exact same output

.. code:: python

    spikes, state = cell(data, state)

*Note*: This is similar to PyTorch's `RNN module <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN>`_ 
if you are looking for inspiration.

Using Norse neurons with time
================================
Similar to PyTorch's `Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_,
Norse's neuron models can be chained in a network.
Unfortunately, this does not work with neurons for the same reason that it does
not work with PyTorch's own RNNs: state.
Instead, Norse offers a `SequentialState <https://norse.github.io/norse/auto_api/norse.torch.module.sequential.html>`_ 
module that ties stateful modules together:

.. code:: python

    import torch
    import norse

    model = SequentialState(
        torch.nn.Linear(10, 5),
        norse.torch.LIFCell(),
        torch.nn.Linear(5, 1)
    )

    data = torch.ones(8, 10) # (batch, input)
    out, state = model(data) # (8, 1) output shape

Using recurrence in Norse
=========================

All neuron modules have :code:`Cell` and :code:`RecurrentCell` classes. 
The :code:`Cell` class applied above simply works as a feed-forward activation
of the neuron, while the :code:`RecurrentCell` also contains linear and 
recurrent weights (we are weighing both the input *and* the recurrent spikes).
For that reason, we need to inform the module what **shape** it needs to take,
since we have to initialize weights to match the desired input/output shape.

The :code:`RecurrentCell` classes work out of the box and can be plugged
directly into the above code. Note that we have the same input/output shape,
but that it could easily be different:

.. code:: python

    import torch
    import norse

    model = SequentialState(
        torch.nn.Linear(10, 5),
        norse.torch.LIFRecurrentCell(5, 5),
        torch.nn.Linear(5, 1)
    )

    data = torch.ones(8, 10) # (batch, input)
    out, state = model(data) # (8, 1) output shape

You can do the same for other neuron types like the 
`LSNN <https://norse.github.io/norse/auto_api/norse.torch.module.lsnn.html>`_, 
`LIFAdEx <https://norse.github.io/norse/auto_api/norse.torch.module.lif_adex.html>`_, etc. 

Using Norse in time
===================

The above ``XCell``s follow the abstraction from PyTorch where the cells are "simple"
activation functions that is applied once.
However, neurons exist in time and will need to be given at least a few timesteps of
input before something interesting happens (like a spike).

The network above (the one without time) works perfectly well with time, and you can
easily wrap it with a for loop. However, it's also possible to run each module
individually in time.

In Norse, we model this time aspect by removing the :code:`Cell` suffix from
the model. So the a :code:`LIFCell` in time will simply be called :code:`LIF`.
Similarly, a :code:`LIFRecurrentCell` in time will simply be called :code:`LIFRecurrent`.

The regular Torch modules also need to run in time. For that, we
added a module to **lift** PyTorch modules into the time domain (that is,
simply run them once for every timestep).

Taken together, we get the following:

.. code:: python

    import torch
    import norse

    model = SequentialState(
        norse.Lift(torch.nn.Linear(10, 5)),
        norse.LSNNRecurrent(5, 5),
        norse.Lift(torch.nn.Linear(5, 1))
    )
    data = torch.ones(100, 8, 10) # (time, batch, input)
    out, state = model(data)

This covers the most basic way to apply Norse. More information can be found
:ref:`page-spiking`, :ref:`page-working` and :ref:`page-spike-learning`.
