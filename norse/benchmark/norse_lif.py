import time
import torch

from norse.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParameters,
    lif_feed_forward_step,
)
from norse.torch.module.encode import PoissonEncoder

from benchmark import BenchmarkParameters


def lif_feed_forward_benchmark(parameters: BenchmarkParameters):
    fc = torch.nn.Linear(parameters.features, parameters.features, bias=False).to(
        parameters.device
    )
    T = parameters.sequence_length
    s = LIFFeedForwardState(
        v=torch.zeros(parameters.batch_size, parameters.features).to(parameters.device),
        i=torch.zeros(parameters.batch_size, parameters.features).to(parameters.device),
    )
    p = LIFParameters(alpha=100.0, method="heaviside")
    input_spikes = PoissonEncoder(T, dt=parameters.dt)(
        0.3
        * torch.ones(
            parameters.batch_size, parameters.features, device=parameters.device
        )
    )
    start = time.time()

    spikes = []
    for ts in range(T):
        x = fc(input_spikes[ts, :])
        z, s = lif_feed_forward_step(input_tensor=x, state=s, parameters=p, dt=0.001)
        spikes += [z]

    spikes = torch.stack(spikes)
    end = time.time()
    duration = end - start
    return duration
