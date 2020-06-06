import time
import torch

from pysnn.connection import Linear
from pysnn.encoding import PoissonEncoder
from pysnn.neuron import LIFNeuron, Input

from benchmark import BenchmarkParameters


def lif_feed_forward_benchmark(parameters: BenchmarkParameters):
    T = int(parameters.dt * parameters.sequence_length)

    # Default parameters from Norse neuron model
    tau_syn = torch.as_tensor(5e-3)
    tau_mem = torch.as_tensor(1e-2)
    v_th = torch.as_tensor(1.0)
    v_reset = torch.as_tensor(0.0)

    input_layer = Input(
        (parameters.batch_size, 1, parameters.features),
        dt=parameters.dt,
        alpha_t=1.0,
        tau_t=1.0,
    ).to(parameters.device)
    linear_layer = Linear(
        in_features=parameters.features,
        out_features=parameters.features,
        batch_size=parameters.batch_size,
        dt=parameters.dt,
        delay=0,
    ).to(parameters.device)
    lif_layer = LIFNeuron(
        cells_shape=(parameters.batch_size, 1, parameters.features),
        thresh=v_th,
        v_rest=v_reset,
        alpha_v=1.0,
        alpha_t=1.0,
        dt=parameters.dt,
        duration_refrac=0.001,
        tau_v=tau_syn,
        tau_t=tau_mem,
        update_type="exponential",
    ).to(parameters.device)

    input_spikes = (
        PoissonEncoder(duration=T, dt=parameters.dt)(
            0.3 * torch.ones(parameters.batch_size, parameters.features)
        )
        .reshape(
            parameters.batch_size, 1, parameters.features, parameters.sequence_length
        )
        .to(parameters.device)
    )

    start = time.time()
    spikes = []
    for ts in range(parameters.sequence_length):
        z, t = input_layer(input_spikes[:, :, :, ts])
        z, _ = linear_layer(z, t)
        z, _ = lif_layer(z)
        spikes += [z]

    spikes = torch.stack(spikes)
    end = time.time()

    duration = end - start
    return duration
