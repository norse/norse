.. _page-installing:

Installing Norse
-------------------

We have chosen to build Norse with new features such as `type hints <https://docs.python.org/3/whatsnew/3.7.html#whatsnew37-pep560>`_. For
that reason we require Python version 3.7 or higher. 
If this is a problem, it is recommended to install Norse from a Docker image using the correct Python version.

Norse builds on top of the `PyTorch <https://pytorch.org/>`_ deep learning library, which is also our
primary dependency.
This has the benefit that your models are hardware accelerated, providing the 
`prerequisites are met <https://pytorch.org/get-started/locally/>`_.

Required dependencies
=====================

Because we are relying on optimised C++ for some of the hotspots in the library, you will need
to download and install `CMake <https://cmake.org/>`_ and `PyTorch <https://pytorch.org/get-started/locally/>`_
*before* you can install Norse.
For that reason, we recommend `following the PyTorch "Get Started" guide <https://pytorch.org/get-started/locally/>`_ as the first step.

Installation steps
==================

This requires command-line access to 

.. code:: bash

    pip install norse

* Installing from Conda

.. code:: bash

    conda install -c norse norse

* Installing from source
 
.. code:: bash

    git clone https://github.com/norse/norse
    cd norse
    python setup.py install



Optional dependencies
=====================

Some of the tasks require additional dependencies like 
`Torchtext <https://pytorch.org/text/stable/index.html>`_ and 
`Torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`_.
We also offer support for `Tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ 
to make it easier to visualise the training and introspect models.
