.. page-index:

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

.. image:: https://img.shields.io/pypi/v/norse
   :target: https://pypi.org/project/norse/

.. image:: https://img.shields.io/conda/v/norse/norse
   :target: https://anaconda.org/norse

.. image:: https://img.shields.io/discord/723215296399147089
   :target: https://discord.gg/7fGN359

.. image:: https://app.codacy.com/project/badge/Grade/a9ab846fc6114afda4320badcb8a69c2
   :target: https://www.codacy.com/gh/norse/norse?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=norse/norse&amp;utm_campaign=Badge_Grade

.. image:: https://codecov.io/gh/norse/norse/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/norse/norse

The purpose of this library is to exploit the advantages of `bio-inspired neural components <https://en.wikipedia.org/wiki/Spiking_neural_network>`_, who are sparse and event-driven - a fundamental difference from artificial neural networks.
Norse expands on `PyTorch <https://pytorch.org/>`_, bringing you two advantages: a modern and proven infrastructure and deep learning-compatible components.

Read more in the :ref:`page-spiking` and :ref:`page-working`.
 
Getting started
-------------------

To try Norse, the best option is to run our `Jupyter Notebook Tutorials <https://github.com/norse/notebooks/>`_ online. 

Alternatively, install Norse and run one of the `included tasks <https://norse.github.io/norse/tasks.html>`_ such as 
`MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_:

.. code:: bash

    python -m norse.task.mnist

The :ref:`page-started` and :ref:`page-working` pages show how to build your own models with Norse while explaining a few
fundamental concepts around spiking neural networks.

Installing Norse
----------------------

Note that we assume you are using Python version 3.7+, are in a terminal friendly environment, and have installed the necessary requirements, 
depending on your installation method. 
More detailed installation instructions are available here: :ref:`page-installing`.

.. role:: bash(code)
    :language: bash

======================== =================================================================================== ====================
  Method                  Instructions                                                                        Prerequisite
------------------------ ----------------------------------------------------------------------------------- --------------------
**From PyPi**             :bash:`pip install norse`                                                          `Pip <https://pypi.org/>`_
**From Conda**            :bash:`conda install -c norse norse`                                               `Conda <https://docs.anaconda.com/anaconda/install/>`_
**From Source**           :bash:`git clone https://github.com/norse/norse && python norse/setup.py install`  `Python <https://www.python.org/>`_
**With Docker**           :bash:`docker pull quay.io/norse/norse`                                            `Docker <https://www.docker.com/get-started>`_
======================== =================================================================================== ====================

Running Tasks
-------------

Norse comes with a number of example tasks, serving as short, self contained, correct examples `SSCCE <http://www.sscce.org/>`_.
They can be run by invoking the ``norse`` module from the base directory.
More information and tasks are available at :ref:`page-tasks` and in your console by typing: :bash:`python -m norse.task.<task> --help`, 
where ``<task>`` is one of the task names.

.. code:: bash

    python3 -m norse.task.mnist

Read more in our :ref:`page-spiking` and visit our `Jupyter Notebook examples <https://github.com/norse/notebooks>`_. 

Advanced uses and opimizations
------------------------------

Norse is meant to be used as a library for spiking neural networks in customized deep learning models.
This typically means porting other models to the spiking/temporal domain, 
extending existing models, 
or starting completely from scratch. 
All three use cases are motivated and briefly described in :ref:`page-working`.

Unfortunately, spiking neural networks are resource intensive.
The page on :ref:`page-hardware` explains how to accelerate the simulations using dedicated hardware.

About Norse
-----------

Norse is a research project created by Christian Pehle and Jens E. Pedersen.
Read more about why we created Norse in :ref:`page-about`.

Table of contents
----------------------------

.. toctree::
   :maxdepth: 2
   :caption: First steps
   :numbered:

   ∇ Home <index>
   installing
   started
   tasks

.. toctree::
   :maxdepth: 2
   :caption: Usage docs
   :numbered:

   about
   hardware
   spiking
   learning
   papers
   working

.. toctree::
   :caption: API Docs
   :glob:
   :titlesonly:

   auto_api/norse.benchmark
   auto_api/norse.task
   auto_api/norse.torch



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
