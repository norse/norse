import time
import torch

from norse.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParameters,
    lif_feed_forward_step,
)
from norse.torch.module.encode import PoissonEncoder

from benchmark import BenchmarkParameters

def no_grad(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def lif_feed_forward_benchmark(parameters: BenchmarkParameters):
    fc = torch.nn.Linear(parameters.features, parameters.features, bias=False).to(
        parameters.device
    )
    no_grad(fc)
    T = parameters.sequence_length
    p = LIFParameters(method="heaviside")
    s = LIFFeedForwardState(
        v=p.v_leak,
        i=torch.zeros(
            parameters.batch_size, parameters.features, device=parameters.device, requires_grad=False
        ),
    )
    input_spikes = PoissonEncoder(T, dt=parameters.dt)(
        0.3
        * torch.ones(
            parameters.batch_size, parameters.features, device=parameters.device
        )
    )
    input_spikes.requires_grad = False
    start = time.time()

    spikes = []
    for ts in range(T):
        x = fc(input_spikes[ts, :])
        z, s = lif_feed_forward_step(input_tensor=x, state=s, p=p, dt=parameters.dt)
        spikes.append(z)

    spikes = torch.stack(spikes)
    end = time.time()
    duration = end - start
    return duration
