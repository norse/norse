import pandas as pd
import time
from typing import Callable
import torch

from norse.torch.functional.encode import poisson_encode
from norse.benchmark.benchmark import BenchmarkParameters
from norse.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParametersJIT,
    _lif_feed_forward_step_jit,
    lif_feed_forward_step,
    lif_feed_forward_step_sparse,
)
from norse.torch.functional.adjoint.lif_adjoint import (
    lif_feed_forward_adjoint_step_sparse,
)


def lif_integrate_adjoint_sparse(
    input_spikes: torch.Tensor,
    p: LIFParametersJIT,
    s: LIFFeedForwardState,
    w: torch.Tensor,
) -> torch.Tensor:
    timesteps = input_spikes.shape[0]
    outs = []
    for ts in range(timesteps):
        current = torch.sparse.mm(input_spikes[ts], w)
        z_out, s = lif_feed_forward_adjoint_step_sparse(current, s, p)
        outs.append(z_out)
    return torch.stack(outs[1:])


def lif_integrate_sparse(
    input_spikes: torch.Tensor,
    p: LIFParametersJIT,
    s: LIFFeedForwardState,
    w: torch.Tensor,
) -> torch.Tensor:
    timesteps = input_spikes.shape[0]
    outs = []
    for ts in range(timesteps):
        current = torch.sparse.mm(input_spikes[ts], w).coalesce()
        z_out, s = lif_feed_forward_step_sparse(current, s, p)
        outs.append(z_out)
    return torch.stack(outs[1:])


def lif_integrate_dense(
    input_spikes: torch.Tensor,
    p: LIFParametersJIT,
    s: LIFFeedForwardState,
    w: torch.Tensor,
) -> torch.Tensor:
    timesteps = input_spikes.shape[0]
    # spikes = torch.jit.annotate(List[Tensor], [])
    spikes = torch.empty(input_spikes.shape, device=input_spikes.device)
    for ts in range(timesteps):
        current = torch.mm(input_spikes[ts], w)
        z_out, s = lif_feed_forward_step(current, s, p)
        spikes[ts] = z_out
    return spikes


@torch.jit.script
def lif_integrate_dense_jit(
    input_spikes: torch.Tensor,
    p: LIFParametersJIT,
    s: LIFFeedForwardState,
    w: torch.Tensor,
) -> torch.Tensor:
    timesteps = input_spikes.shape[0]
    # spikes = torch.jit.annotate(List[Tensor], [])
    spikes = torch.empty(input_spikes.shape, device=input_spikes.device)
    for ts in range(timesteps):
        current = torch.mm(input_spikes[ts], w)
        z_out, s = _lif_feed_forward_step_jit(current, s, p)
        spikes[ts] = z_out
    return spikes


def _benchmark(model: Callable, parameters: BenchmarkParameters, sparse: bool):
    def _copy_sparse(other, value):
        other = other.coalesce()
        return torch.sparse_coo_tensor(
            indices=other.indices(),
            values=torch.full_like(other.values(), value),
            size=other.size(),
            device=other.device,
        ).coalesce()

    input_spikes = (
        poisson_encode(
            torch.ones(
                parameters.batch_size, parameters.features, device=parameters.device
            ),
            seq_length=parameters.timesteps,
            f_max=parameters.poisson_rate,
            dt=parameters.dt,
        )
        .requires_grad_(True)
        .to(parameters.device)
    )
    w = (
        torch.randn(parameters.features, parameters.features)
        .requires_grad_(True)
        .to(parameters.device)
    )
    p = LIFParametersJIT(
        tau_mem_inv=torch.tensor(1 / 100),
        tau_syn_inv=torch.tensor(1 / 50),
        v_leak=torch.tensor(0.0),
        v_th=torch.tensor(1.0),
        v_reset=torch.tensor(0.0),
        method="super",
        alpha=torch.tensor(100.0),
    )
    s = LIFFeedForwardState(
        v=p.v_leak.to(parameters.device),
        i=torch.zeros(
            parameters.batch_size,
            parameters.features,
            device=parameters.device,
        ),
    )

    if sparse:
        input_spikes = input_spikes.to_sparse().coalesce()
        w = w.to_sparse().coalesce()
        if sparse:
            p = LIFParametersJIT(
                tau_mem_inv=torch.tensor(1 / 100),
                tau_syn_inv=torch.tensor(1 / 50),
                v_leak=_copy_sparse(input_spikes[0], 0),
                v_th=_copy_sparse(input_spikes[0], 1),
                v_reset=_copy_sparse(input_spikes[0], 0),
                method="super",
                alpha=torch.tensor(100.0),
            )
            s = LIFFeedForwardState(
                v=p.v_leak.to(parameters.device),
                i=torch.zeros(
                    parameters.batch_size,
                    parameters.features,
                    device=parameters.device,
                ).to_sparse(),
            )
    else:
        input_spikes = input_spikes.contiguous()
        w = w.contiguous()

    start = time.time()
    z_out = model(input_spikes, p, s, w)
    end = time.time()
    duration_ff = end - start
    if sparse or model == lif_integrate_adjoint_sparse:
        loss = torch.sparse.sum(z_out)
    else:
        loss = torch.sum(z_out)
    start = time.time()
    loss.backward()
    end = time.time()
    duration_fb = end - start
    return pd.DataFrame({"forward": [duration_ff], "backward": [duration_fb]})


def benchmark_feedforward_adjoint_sparse(parameters: BenchmarkParameters):
    return _benchmark(lif_integrate_adjoint_sparse, parameters, sparse=False)


def benchmark_feedforward_dense(parameters: BenchmarkParameters):
    return _benchmark(lif_integrate_dense, parameters, sparse=False)


def benchmark_feedforward_dense_jit(parameters: BenchmarkParameters):
    return _benchmark(lif_integrate_dense_jit, parameters, sparse=False)


def benchmark_feedforward_sparse(parameters: BenchmarkParameters):
    return _benchmark(lif_integrate_sparse, parameters, sparse=True)


if __name__ == "__main__":
    p = BenchmarkParameters(
        32, "cuda", dt=0.001, features=16, poisson_rate=0.01, timesteps=1024
    )
    out = benchmark_feedforward_adjoint_sparse(p)
    out = benchmark_feedforward_dense(p)
    out = benchmark_feedforward_dense_jit(p)
    out = benchmark_feedforward_sparse(p)
    out = benchmark_feedforward_sparse(p)
    print(out)
