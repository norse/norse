import time
import torch
import traceback

from lava.lib.dl.slayer.neuron import cuba
from norse.torch.module.encode import PoissonEncoder

# pytype: disable=import-error
from ..benchmark import BenchmarkParameters

# pytype: enable=import-error


class LIFBenchmark(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.fc = torch.nn.Linear(parameters.features, parameters.features, bias=False)
        self.neuron = cuba.Neuron(
            threshold=1.0,
            current_decay=0.9,
            voltage_decay=0.9,
            persistent_state=False,
            shared_param=False,
        )

    def forward(self, input_spikes: torch.Tensor):
        x = self.fc(input_spikes)
        x = x.permute(1, 2, 0)  # BNT
        return self.neuron(x)


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
            _ = model(input_sequence)

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
