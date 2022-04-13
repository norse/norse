import time
import torch

from norse.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParameters,
    _lif_feed_forward_integral_jit,
)
from norse.torch.module.encode import PoissonEncoder
from norse.torch.module import LIFCellCG
from norse.torch.module import LIFCell

# pytype: disable=import-error
from benchmark import BenchmarkParameters

# pytype: enable=import-error


class LIFBenchmark(torch.nn.Module):
    def __init__(self, parameters, p: LIFParameters):
        super().__init__()
        self.fc = torch.nn.Linear(parameters.features, parameters.features, bias=False)
        self.lif = LIFCell(p=p)
        self.dt = parameters.dt

    def forward(self, input_spikes: torch.Tensor):
        seq_len, batch_size, _ = input_spikes.shape

        state = None

        for t_step in range(seq_len):
            input = input_spikes[t_step, :, :]
            z, state = self.lif(self.fc(input), state) 

        return z, state


class LIFCGBenchmark(torch.nn.Module):
    def __init__(self, parameters, p: LIFParameters):
        super().__init__()
        self.fc = torch.nn.Linear(parameters.features, parameters.features, bias=False)
        self.lif_cg = LIFCellCG(p=p)
        self.dt = parameters.dt

    def forward(self, input_spikes: torch.Tensor):

        seq_len, batch_size, _ = input_spikes.shape

        state = None

        for t_step in range(seq_len):
            input = input_spikes[t_step, :, :]
            z, state = self.lif_cg(self.fc(input), state) 

        return z, state
        

def lif_cell_benchmark(parameters: BenchmarkParameters):
    with torch.no_grad():
        input_sequence = torch.randn(
            parameters.sequence_length,
            parameters.batch_size,
            parameters.features,
            device=parameters.device,
        )
        p = LIFParameters()
        model = LIFBenchmark(parameters, p).to(parameters.device)

        # Set real data
        poisson_data = PoissonEncoder(parameters.sequence_length, dt=parameters.dt)(0.3 * torch.ones(parameters.batch_size, parameters.features, device=parameters.device)).contiguous()
        input_sequence.copy_(poisson_data)

        # Start recording
        start = time.time()
        zo, so = model(input_sequence)
        end = time.time()
        duration = end - start
        return duration


def lif_cell_cg_benchmark(parameters: BenchmarkParameters):
    with torch.no_grad():
        input_sequence = torch.randn(
            parameters.sequence_length,
            parameters.batch_size,
            parameters.features,
            device=parameters.device,
        )
        p = LIFParameters()
        model = LIFCGBenchmark(parameters, p).to(parameters.device)

        # Set real data
        poisson_data = PoissonEncoder(parameters.sequence_length, dt=parameters.dt)(0.3 * torch.ones(parameters.batch_size, parameters.features, device=parameters.device)).contiguous()
        input_sequence.copy_(poisson_data)

        # Start recording
        start = time.time()
        zo, so = model(input_sequence)
        end = time.time()
        duration = end - start
        return duration