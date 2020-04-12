.. _page-introduction-spiking:

Introduction to spiking neural networks
===================================

Regular neural networks consist of atomic neurons, that basically take a lot of input and sends it through a function (typically sigmoidal).
The issue with this approach is that the temporal dimension is mostly ignored. 
In biology, neural networks are living things that change over time and where sparseness plays a critical role. 
Norse has shown how it is feasible to achieve state-of-the-art results **and** the desirable properties of spiking neural networks: low energy consumption and speed.

We have prepared examples and tutorials that will familiarise you with SNNs and give you a grasp on their sparse nature. 

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   auto_examples/index
   experiments