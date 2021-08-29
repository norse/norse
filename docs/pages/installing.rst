.. _page-installing:

Installing Norse
----------------

We have chosen to build Norse with new features such as `type hints <https://docs.python.org/3/whatsnew/3.7.html#whatsnew37-pep560>`_. For
that reason **we require Python version 3.7 or higher**. 
If this is a problem, it is recommended to install Norse from a 
`Docker <https://en.wikipedia.org/wiki/Docker_(software)>`_ image.

Norse builds on top of the `PyTorch <https://pytorch.org/>`_ deep learning library, which is also our
primary dependency.
This has the benefit that your models are hardware accelerated, providing the 
`prerequisites are met <https://pytorch.org/get-started/locally/>`_.

Required dependencies
=====================

Because we are relying on optimised C++ for some of the hotspots in the library, you will need
to download and install  `CMake <https://cmake.org/>`_ and `PyTorch <https://pytorch.org/get-started/locally/>`_
*before* you can install Norse.
For that reason, we recommend `following the PyTorch "Get Started" guide <https://pytorch.org/get-started/locally/>`_ as the first step.

You might also have to install Python headers if you have not already done that. In Debian-based distros (like Ubuntu),
this can be done by running ``apt install python3-dev``.

Installation steps
==================

Note that the following commands require access to a 
`command line interface <https://en.wikipedia.org/wiki/Command-line_interface>`_.

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

* Installing with `Docker <https://en.wikipedia.org/wiki/Docker_(software)>`_

.. code:: bash
    
    docker pull quay.io/norse/norse


Windows
^^^^^^^

We strongly recommend you use the Docker approach when installing Norse on Windows.
Please see `the installation guide for Windows <https://docs.docker.com/docker-for-windows/install/>`_ for accurate instructions. 

We also provide a Docker image bundled with Conda, available like so: 
``docker pull quay.io/norse/norse:conda-latest``.


Optional dependencies
=====================

Some of the tasks require additional dependencies like 
`Pytorch Lightning <https://pytorchlightning.ai/>`_,
`Torchtext <https://pytorch.org/text/stable/index.html>`_ and 
`Torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`_.
We also offer support for `Tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ 
to make it easier to visualise the training and introspect models.


Running Norse notebooks with Docker
===================================

Docker creates a closed environment for you, which also means that the network and
filesystem is isolated. Without going into details, here are three steps you can
take to create a `Jupyter Notebook <https://jupyter.org/>`_ environment with
Docker. You will have to replace ``/your/directory`` with the **full** path to
your current directory.

.. code:: bash

    docker run -it -p 8888:8888 -v /your/directory:/work quay.io/norse/norse bash
    pip3 install jupyter
    jupyter notebook --notebook-dir=/work --ip 0.0.0.0 --allow-root

The command line will now show you a URL you can copy-paste into your browser.
And voila!

GPU acceleration
^^^^^^^^^^^^^^^^

If you would like to have GPU hardware acceleration, you will have to enable the
NVIDIA runtime, as described here: https://developer.nvidia.com/nvidia-container-runtime.

For more information, please refer to our page on :ref:`page-hardware`.