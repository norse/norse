import time
import torch

from norse.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParameters,
    _lif_feed_forward_integral_jit,
)
from norse.torch.module.encode import PoissonEncoder

# pytype: disable=import-error
from .benchmark import BenchmarkParameters

# pytype: enable=import-error


class LIFBenchmark(torch.jit.ScriptModule):
    def __init__(self, parameters):
        super().__init__()
        self.fc = torch.nn.Linear(parameters.features, parameters.features, bias=False)
        self.dt = parameters.dt

    def forward(
        self, input_spikes: torch.Tensor, p: LIFParameters, s: LIFFeedForwardState
    ):
        x = self.fc(input_spikes)
        return _lif_feed_forward_integral_jit(input_tensor=x, state=s, p=p, dt=self.dt)


def lif_feed_forward_benchmark(parameters: BenchmarkParameters):
    with torch.no_grad():
        model = LIFBenchmark(parameters).to(parameters.device)
        input_sequence = torch.randn(
            parameters.sequence_length,
            parameters.batch_size,
            parameters.features,
            device=parameters.device,
        )
        p = LIFParameters()
        s = LIFFeedForwardState(
            v=torch.full(
                (parameters.batch_size, parameters.features),
                p.v_leak,
                device=parameters.device,
            ),
            i=torch.zeros(
                parameters.batch_size,
                parameters.features,
                device=parameters.device,
            ),
        )

        # Warmup cuda stream
        g = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                _ = model(input_sequence, p, s)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(g):
            _ = model(input_sequence, p, s)

        # Set real data
        poisson_data = PoissonEncoder(parameters.sequence_length, dt=parameters.dt)(
            0.3
            * torch.ones(
                parameters.batch_size, parameters.features, device=parameters.device
            )
        ).contiguous()
        input_sequence.copy_(poisson_data)

        # Start recording
        start = time.time()
        g.replay()
        end = time.time()
        duration = end - start
        return duration
