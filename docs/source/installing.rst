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

For that reason, we recommend `following the PyTorch "Get Started" guide <https://pytorch.org/get-started/locally/>`_ as the first step.

After installing the Torch prerequisite, the primary dependencies can be done in the following three ways:

Installing from PyPi
====================

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
    pip install -e norse


Optional dependencies
=====================

Norse runs well with the above installation, but some of the tasks integrates with the visualization
tool `Tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ to make it easier to
introspect models and results.
