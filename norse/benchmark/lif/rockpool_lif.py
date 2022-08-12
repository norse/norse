import time
import torch

from rockpool.nn.modules import lif_neuron_torch
from norse.torch.module.encode import PoissonEncoder

# pytype: disable=import-error
from ..benchmark import BenchmarkParameters

# pytype: enable=import-error


class LIFBenchmark(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.fc = torch.nn.Linear(parameters.features, parameters.features, bias=False)
        self.neuron = lif_neuron_torch.LIFNeuronTorch(
            shape=(parameters.features,), dt=parameters.dt
        ).to(parameters.device)

    def forward(
        self,
        input_spikes: torch.Tensor,
    ):
        x = self.fc(input_spikes)
        return self.neuron(x)


def lif_feed_forward_benchmark(parameters: BenchmarkParameters):
    with torch.no_grad():
        model = LIFBenchmark(parameters).to(parameters.device)
        input_sequence = torch.randn(
            parameters.batch_size,
            parameters.sequence_length,
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
        poisson_data = poisson_data.permute(1, 0, 2)  # Reshape to BTN

        # Start recording
        start = time.time()
        model(poisson_data)
        end = time.time()
        duration = end - start
        return duration
