import time
import torch

from norse.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParametersJIT,
    _lif_feed_forward_step_jit,
)
from norse.torch.module.encode import PoissonEncoder

from benchmark import BenchmarkParameters


class LIFBenchmark(torch.jit.ScriptModule):
    def __init__(self, parameters):
        super(LIFBenchmark, self).__init__()
        self.fc = torch.nn.Linear(parameters.features, parameters.features, bias=False)
        self.dt = parameters.dt

    @torch.jit.script_method
    def forward(
        self, input_spikes: torch.Tensor, p: LIFParametersJIT, s: LIFFeedForwardState
    ):
        sequence_length, batch_size, features = input_spikes.shape

        inputs = input_spikes.unbind(0)
        # spikes = torch.jit.annotate(List[Tensor], [])
        spikes = torch.empty((sequence_length, batch_size, features))

        for ts in range(len(inputs)):
            x = self.fc(inputs[ts])
            z, s = _lif_feed_forward_step_jit(input_tensor=x, state=s, p=p, dt=self.dt)
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
        p = LIFParametersJIT(
            tau_syn_inv=torch.as_tensor(1.0 / 5e-3),
            tau_mem_inv=torch.as_tensor(1.0 / 1e-2),
            v_leak=torch.as_tensor(0.0),
            v_th=torch.as_tensor(1.0),
            v_reset=torch.as_tensor(0.0),
            method="super",
            alpha=torch.as_tensor(0.0),
        )
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
