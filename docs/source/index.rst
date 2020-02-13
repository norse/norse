.. norse documentation master file, created by
   sphinx-quickstart on Mon Jun 24 16:20:44 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Norse
=====

Norse is a library to do 
`deep learning <https://en.wikipedia.org/wiki/Deep_learning>`_ 
with 
`spiking neural networks <https://en.wikipedia.org/wiki/Spiking_neural_network>`_.

.. |image:: https://github.com/norse/norse/workflows/Python%20package/badge.svg
   :target: https://github.com/norse/norse/actions

The purpose of this library is to exploit the advantages of `bio-inspired neural components <https://en.wikipedia.org/wiki/Spiking_neural_network>`_, who are sparse and event-driven - a fundamental difference from artificial neural networks.
Norse expands `PyTorch <https://pytorch.org/>`_ with primitives for bio-inspired neural components, 
bringing you two advantages: a modern and proven infrastructure based on PyTorch and deep learning-compatible spiking neural network components.
 
Getting started
---------------

Norse is a library that you can either install via `PyPi <https://pypi.org/project/norse/>`_ or from source. Note that `pip <https://pip.pypa.io/en/stable/>`_ is required.

* Installing from PyPi: 

.. code:: bash

    pip install norse

* Installing from source:
 
.. code:: bash

    git clone https://github.com/norse/norse
    pip install -e norse

Running experiments
-------------------

Running experiments is as easy as running 

.. code:: bash

    python3 -m norse

This will run a simple MNIST experiment with default parameters. See :doc:`/experiments` for detailed instructions.

Table of contents
----------------------------

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   auto_examples/index
   experiments

.. toctree::
   :maxdepth: 2
   :caption: API Docs
   :glob:
   :titlesonly:
   
   functional
   module



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
