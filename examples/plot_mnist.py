# -*- coding: utf-8 -*-
r"""
Spiking Neural Networks
=======================

Spiking neural networks are not that much different than Artificial Neural
Networks that are currently most commonly in use. The main difference is
that there is an insistence that communication between units in the network
happens through spikes - represented as binary one or zero - and that time
is involved.

How to define a Network
-----------------------

The spiking neural network primitives in myelin are designed to fit in as seamlessly
as possible into a traditional deep learning pipeline.
"""

import torch
from myelin.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParameters,
    lif_feed_forward_step,
)
from myelin.torch.functional.leaky_integrator import LIState

from myelin.torch.module.leaky_integrator import LICell
from myelin.torch.module.lif import LIFFeedForwardCell


class Net(torch.nn.Module):
    def __init__(
        self,
        device="cpu",
        num_channels=1,
        feature_size=32,
        model="super",
        dtype=torch.float,
    ):
        super(Net, self).__init__()
        self.features = int(((feature_size - 4) / 2 - 4) / 2)

        self.conv1 = torch.nn.Conv2d(num_channels, 32, 5, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1)
        self.fc1 = torch.nn.Linear(self.features * self.features * 64, 1024)
        self.lif0 = LIFFeedForwardCell(
            (32, feature_size - 4, feature_size - 4),
            p=LIFParameters(method=model, alpha=100.0),
        )
        self.lif1 = LIFFeedForwardCell(
            (64, int((feature_size - 4) / 2) - 4, int((feature_size - 4) / 2) - 4),
            p=LIFParameters(method=model, alpha=100.0),
        )
        self.lif2 = LIFFeedForwardCell(
            (1024,), p=LIFParameters(method=model, alpha=100.0)
        )
        self.out = LICell(1024, 10)
        self.device = device
        self.dtype = dtype

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0 = self.lif0.initial_state(batch_size, device=self.device, dtype=self.dtype)
        s1 = self.lif1.initial_state(batch_size, device=self.device, dtype=self.dtype)
        s2 = self.lif2.initial_state(batch_size, device=self.device, dtype=self.dtype)
        so = self.out.initial_state(batch_size, device=self.device, dtype=self.dtype)

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=self.device, dtype=self.dtype
        )

        for ts in range(seq_length):
            z = self.conv1(x[ts, :])
            z, s0 = self.lif0(z, s0)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = z.view(-1, self.features ** 2 * 64)
            z = self.fc1(z)
            z, s2 = self.lif2(z, s2)
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[ts, :, :] = v
        return voltages


net = Net()
print(net)

########################################################################
# We can evaluate the network we just defined on an input of size 1x32x32.
# Note that in contrast to typical spicing neural network simulators time
# is just another dimension in the input tensor here we chose to evaluate
# the network on 16 timesteps and there is an explicite batch dimension
# (number of concurrently evaluated inputs with identical model parameters).

timesteps = 16
batch_size = 1
input = torch.abs(torch.randn(timesteps, batch_size, 1, 32, 32))
out = net(input)
print(out)


##########################################################################
# Since the spiking neural network is implemented as a pytorch module, we  
# can use the usual pytorch primitives for optimizing it. Note that the
# backward computation expects a gradient for each timestep

net.zero_grad()
out.backward(torch.randn(timesteps, batch_size, 10))

########################################################################
# .. note::
#
#     ``myelin`` like pytorch only supports mini-batches. This means that
#     contrary to most other spiking neural network simulators ```myelin``` always
#     integrates several indepdentent sets of spiking neural networks at once.
