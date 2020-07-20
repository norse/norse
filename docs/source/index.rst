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

.. image:: https://github.com/norse/norse/workflows/Python%20package/badge.svg
   :target: https://github.com/norse/norse/actions

The purpose of this library is to exploit the advantages of `bio-inspired neural components <https://en.wikipedia.org/wiki/Spiking_neural_network>`_, who are sparse and event-driven - a fundamental difference from artificial neural networks.
Norse expands on `PyTorch <https://pytorch.org/>`_, bringing you two advantages: a modern and proven infrastructure and deep learning-compatible components.

Read more in the :ref:`page-spiking` and :ref:`page-working`.
 
Getting started
---------------

To try Norse, the best option is to run one of the `Jupyter Notebooks <https://github.com/norse/notebooks/>`_ on Google collab. 

Alternatively, you can install run one of the :ref:`page-experiments` such as `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_:

.. code:: bash

    python -m norse.task.mnist

Installing Norse
----------------

Note that we assume you are using Python version 3.7+, are in a terminal friendly environment, and have installed the necessary requirements, 
depending on your installation method. 
More detailed installation instructions are available here: :ref:`page-installing`.

* Installing from PyPi

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

Running experiments
-------------------

Norse is bundled with a number of example experiments, serving as short, self contained, correct examples `SSCCE <http://www.sscce.org/>`_.
They can be run by invoking the ``norse`` module from the base directory.
More information and tasks are available at :ref:`page-experiments` and in your console by typing: ``python -m norse.task.<task> --help``, where ``<task>`` is one of the task names.

.. code:: bash

    python3 -m norse.task.mnist

This will run an MNIST experiment with default parameters. 
See :doc:`/experiments` for detailed instructions.

Tutorials and understanding spikes
----------------------------------

In biology, neural networks are living things that change over time and where sparseness plays a critical role. 
We have prepared some introductory texts and tutorials that will help you understand why we use spiking neural networks in our work. 

Read more in our :ref:`page-spiking` and visit our `Jupyter Notebook examples <https://github.com/norse/notebooks>`_. 

Using Norse in your own work
----------------------------

Norse is meant to be used as a library for spiking neural networks in customized deep learning models.
This typically means porting other models to the spiking/temporal domain, 
extending existing models, 
or starting completely from scratch. 
All three use cases are motivated and briefly described in :ref:`page-working`.

About Norse
-----------

Norse is a research project created by Christian Pehle and Jens E. Pedersen.
Read more about why we created Norse in :ref:`page-about`.

Table of contents
----------------------------

.. toctree::
   :maxdepth: 2
   :caption: Usage Docs
   :numbered:

   about
   installing
   spiking
   learning
   working

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :glob:
   :numbered:

   auto_examples/index
   experiments

.. toctree::
   :maxdepth: 2
   :caption: API Docs
   :glob:
   :titlesonly:
   :numbered:

   auto_api/norse.benchmark
   auto_api/norse.module
   auto_api/norse.task
   auto_api/norse.torch



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
