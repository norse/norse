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

# =================================
# Benchmark LIFCell


class LIFBenchmark(torch.nn.Module):
    """
    Simple SNN with one linear layer + LIFCell

    Args:
        parameters (BenchmarkParameters): parameters for benchmarking
        p (LIFParameters): LIFCell parameters
    """

    def __init__(self, parameters, p: LIFParameters):
        super().__init__()
        self.fc = torch.nn.Linear(parameters.features, parameters.features, bias=True)
        self.lif = LIFCell(p=p)

    def forward(self, input_spikes: torch.Tensor):
        """
        Forward pass

        Args:
            input_spikes (torch.Tensor): encoded input spikes

        Returns:
            (Tuple[torch.Tensor, LIFFeedForwardState]): output spikes, current LIFFeedForwardState of LIFCell
        """
        seq_len, batch_size, _ = input_spikes.shape

        state = None

        for t_step in range(seq_len):
            input = input_spikes[t_step, :, :]
            z, state = self.lif(self.fc(input), state)

        return z, state


class LIFCGBenchmark(torch.nn.Module):
    """
    Simple SNN with one linear layer + LIFCell.
    LIFCell exploits CUDA graphs, as described in:
    <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>

    Args:
        parameters (BenchmarkParameters): parameters for benchmarking
        p (LIFParameters): LIFCell parameters
    """

    def __init__(self, parameters, p: LIFParameters):
        super().__init__()
        self.fc = torch.nn.Linear(parameters.features, parameters.features, bias=True)
        self.lif_cg = LIFCellCG(p=p)
        self.dt = parameters.dt

    def forward(self, input_spikes: torch.Tensor):
        """
        Forward pass

        Args:
            input_spikes (torch.Tensor): encoded input spikes

        Returns:
            (Tuple[torch.Tensor, LIFFeedForwardState]): output spikes, current LIFFeedForwardState of LIFCell
        """
        seq_len, batch_size, _ = input_spikes.shape

        state = None

        for t_step in range(seq_len):
            input = input_spikes[t_step, :, :]
            z, state = self.lif_cg(input, state)

        return z, state


def lif_cell_benchmark(parameters: BenchmarkParameters):
    """
    Benchmark LIFCell

    Args:
        parameters (BenchmarkParameters): parameters for benchmarking

    Returns:
        float: duration of simulated dynamics
    """
    p = LIFParameters()
    model = LIFBenchmark(parameters, p).to(parameters.device)

    # Set real data
    encoder = PoissonEncoder(parameters.sequence_length, dt=parameters.dt)
    poisson_data = encoder(
        0.3
        * torch.ones(
            parameters.batch_size, parameters.features, device=parameters.device
        )
    )

    # Start recording
    start = time.time()
    zo, so = model(poisson_data)
    end = time.time()
    duration = end - start
    return duration


def lif_cell_cg_benchmark(parameters: BenchmarkParameters):
    """
    Benchmark LIFCell
    LIFCell exploits CUDA graphs, as described in:
    <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>

    Args:
        parameters (BenchmarkParameters): parameters for benchmarking

    Returns:
        float: duration of simulated dynamics
    """
    p = LIFParameters()
    model = LIFCGBenchmark(parameters, p).to(parameters.device)

    # Set real data
    encoder = PoissonEncoder(parameters.sequence_length, dt=parameters.dt)
    poisson_data = encoder(
        0.3
        * torch.ones(
            parameters.batch_size, parameters.features, device=parameters.device
        )
    )

    # Start recording
    start = time.time()
    zo, so = model(poisson_data)
    end = time.time()
    duration = end - start
    return duration
