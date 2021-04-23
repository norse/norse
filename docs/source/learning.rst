.. _page-spike-learning:

Learning with spikes
------------------------------------------

To be able to learn in spiking neural networks (SNN) one needs to 
update the weights between neurons, as in all types of neural networks.
As you probably know, you need smooth, differentiable functions to 
apply popular algorithms like gradient descent and backpropagation. 

However, spiking neurons do not have smooth activation functions
(a spike either happens or doesn't).
This page aims to explain - in an informan manner - how we can train
SNNs in Norse irregardless of their non-differentiable nature.
The general approach is described in better detail by
`Emre O. Neftci, Hesham Mostafa, and Friedemann Zenke <https://arxiv.org/abs/1901.09948>`_.

Before you read further though, make sure you are familiar 
with how PyTorch 
`works with Autograd and backpropagation <https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#autograd>`_.
Norse is basically applying the same principle.

Training Spiking Neural Networks
=====================================

Many solutions have been attempted to solve this problem with varying 
success. 
Here we will only cover the *surrogate gradient* approach, and
illustrate how  it is implement in norse.

Our approach builds on the SuperSpike method proposed by 
`Steven K. Esser et al. (Convolutional networks for fast, energy-efficient neuromorphic computing) <https://www.pnas.org/content/113/41/11441>`_ and
further elaborated in
`Friedemann Zenke and Surya Ganguli <https://www.mitpressjournals.org/doi/pdf/10.1162/neco_a_01086>`_.

Neurons work in the way that they update their membrane equations with
incoming currents from pre-synaptic neurons. 
If the incoming currents exceed a threshold, the post-synaptic
neuron releases a spike.
Hhis can easily be expressed in code:

.. code::

    if membrane > threshold: spike!

However, what happens when you take the gradient of that? It will be
zero for the most part because ``membrane < threshold``, meaning that
the neuron does not influence the output at all.
But sometimes the current goes above the threshold, you get an
activation, and the gradient changes!
To account for that, we can "pretend" that the gradient doesn't 
have this awkward sudden shift. 
Instead, we can look at the numerical state of the neuron and then
use that as an indicator for how much the neuron influenced the
output.

Given the neuron membrane potential (:math:`U`) and the neuron firing
threshold (:math:`v`), then this is a simplified version of the
SuperSpike surrogate partial derivative for some activation 
function (:math:`\sigma`):

.. math::
    \sigma '(U_i) = \left(1 + |U_i - v| \right)^{-2}

In the SuperSpike algorithm, we look at the *difference* between the 
neuron membrane and the firing threshold.
If, say, the neuron membrane voltage is much higher than the
firing threshold, we know that the neuron `will` fire.
But too far away from that threshold indicates that the contribution
of the neuron is unimportant because it would require a large
modification to that particular neuron to *not* impact the output.

Conversely, if the the neuron membrane voltage is much lower 
than the threshold, the neuron is probably not
going to fire, and the gradient contribution is equally low.

And that's it! SuperSpike permits the calculation of gradients for
non-differentiable functions. 
Which, in turn, permits us to use the native autograd properties
of PyTorch.

An implementation of the SuperSpike algorithm in Norse can be found 
in the
`threshold.py <https://github.com/norse/norse/blob/master/norse/torch/functional/threshold.py>`_
module.
