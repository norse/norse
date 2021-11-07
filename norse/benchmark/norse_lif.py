import time
import torch

from norse.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParameters,
    _lif_feed_forward_step_jit,
    lif_feed_forward_step,
)
from norse.torch.module.encode import PoissonEncoder

# pytype: disable=import-error
from benchmark import BenchmarkParameters

# pytype: enable=import-error


class LIFBenchmark(torch.jit.ScriptModule):
    def __init__(self, parameters):
        super(LIFBenchmark, self).__init__()
        self.fc = torch.nn.Linear(parameters.features, parameters.features, bias=False)
        self.dt = parameters.dt

    def forward(
        self, input_spikes: torch.Tensor, p: LIFParameters, s: LIFFeedForwardState
    ):
        sequence_length, batch_size, features = input_spikes.shape
        # spikes = torch.jit.annotate(List[Tensor], [])
        spikes = torch.empty(
            (sequence_length, batch_size, features), device=input_spikes.device
        )

        for ts in range(sequence_length):
            x = self.fc(input_spikes[ts])
            z, s = lif_feed_forward_step(input_tensor=x, state=s, p=p, dt=self.dt)
            spikes[ts] = z

        return spikes


def lif_feed_forward_benchmark(parameters: BenchmarkParameters):
    with torch.no_grad():
        model = LIFBenchmark(parameters).to(parameters.device)
        input_spikes = PoissonEncoder(parameters.sequence_length, dt=parameters.dt)(
            0.3
            * torch.ones(
                parameters.batch_size, parameters.features, device=parameters.device
            )
        ).contiguous()
        p = LIFParameters()
        s = LIFFeedForwardState(
            v=p.v_leak,
            i=torch.zeros(
                parameters.batch_size,
                parameters.features,
                device=parameters.device,
            ),
        )
        start = time.time()
        model(input_spikes, p, s)
        end = time.time()
        duration = end - start
        return duration
