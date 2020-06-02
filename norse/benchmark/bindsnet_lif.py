import time
import torch

from bindsnet.network import Network
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import LIFNodes, Input
from bindsnet.encoding import PoissonEncoder

from benchmark import BenchmarkParameters


def lif_feed_forward_benchmark(parameters: BenchmarkParameters):
    T = parameters.dt * parameters.sequence_length
    network = Network(batch_size=parameters.batch_size, dt=parameters.dt)

    network.add_layer(Input(n=parameters.features), name="Input")
    network.add_layer(LIFNodes(n=parameters.features), name="Neurons")
    network.add_connection(
        Connection(source=network.layers["Input"], target=network.layers["Neurons"]),
        source="Input",
        target="Neurons",
    )

    input_spikes = (
        PoissonEncoder(time=T, dt=parameters.dt)(
            0.3 * torch.ones(parameters.batch_size, parameters.features)
        )
        .to(parameters.device)
        .float()
    )

    input_data = {"Input": input_spikes}
    network.to(parameters.device)
    start = time.time()
    network.run(inputs=input_data, time=T)
    end = time.time()

    duration = end - start
    return duration
