import time
import torch

from ..benchmark import BenchmarkParameters
from norse.torch import PoissonEncoder
from spikingjelly.activation_based import functional, neuron, surrogate


class LIFBoxBenchmark(torch.nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.lin = torch.nn.Linear(features, features, bias=False)
        self.lif = neuron.LIFNode(
            surrogate_function=surrogate.SoftSign(), step_mode="s"
        )

    def forward(self, input_spikes: torch.Tensor):
        out = []
        self.lif.reset()  # Reset state
        for ts in input_spikes:
            out.append(self.lif(self.lin(ts)))
        return torch.stack(out)


def lif_box_feed_forward_benchmark(parameters: BenchmarkParameters):
    with torch.no_grad():
        model = LIFBoxBenchmark(parameters.features).to(parameters.device)
        input_sequence = torch.randn(
            parameters.sequence_length,
            parameters.batch_size,
            parameters.features,
            device=parameters.device,
        )

        # Warmup cuda stream
        g = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                _ = model(input_sequence)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(g):
            _ = model(input_sequence)

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
