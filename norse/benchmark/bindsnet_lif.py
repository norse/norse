import time
import torch

from bindsnet.network import Network
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import LIFNodes, Input
from bindsnet.encoding import PoissonEncoder
from bindsnet.encoding import poisson

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

    input_data = {
        "Input": poisson(
            datum=torch.rand(
                (parameters.batch_size, parameters.features), device=parameters.device
            ),
            time=T * parameters.sequence_length,
            device=parameters.device,
        )
    }

    network.to(parameters.device)

    with torch.no_grad():
        start = time.time()
        network.run(inputs=input_data, time=T)
        end = time.time()

    duration = end - start
    return duration
