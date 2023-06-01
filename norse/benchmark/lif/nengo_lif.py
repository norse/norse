import time
import torch
import traceback

import numpy as np
import tensorflow
import nengo
import nengo_dl
from norse.torch.module.encode import PoissonEncoder

# pytype: disable=import-error
from ..benchmark import BenchmarkParameters

# pytype: enable=import-error


class LIFBenchmark(torch.nn.Module):
    # Thanks to https://www.nengo.ai/nengo-dl/examples/spiking-mnist.html
    def __init__(self, parameters):
        super().__init__()
        with nengo.Network(seed=0) as net:
            nengo_dl.configure_settings(stateful=False)
            inp = nengo.Node(np.zeros(parameters.features))
            x = nengo_dl.Layer(
                tensorflow.layers.Linear(parameters.features, use_bias=False)
            )(inp)
            x = nengo_dl.Layer(nengo.LIF())(x)
            self.out = nengo.Probe(x)
            self.sim = nengo_dl.Simulator(net, minibatch_size=parameters.batch_size)

    def forward(self, input_spikes: torch.Tensor):
        return sim.predict(input_spikes)


def lif_feed_forward_benchmark(parameters: BenchmarkParameters):
    with torch.no_grad():
        model = LIFBenchmark(parameters).to(parameters.device)
        input_sequence = torch.randn(
            parameters.sequence_length,
            parameters.batch_size,
            parameters.features,
            device=parameters.device,
        )

        # Warmup model
        for _ in range(2):
            _ = model(input_sequence.numpy)

        # Set real data
        poisson_data = PoissonEncoder(parameters.sequence_length, dt=parameters.dt)(
            0.6
            * torch.ones(
                parameters.batch_size, parameters.features, device=parameters.device
            )
        ).contiguous()

        # Start recording
        start = time.time()
        model(poisson_data)
        end = time.time()
        duration = end - start
        return duration
