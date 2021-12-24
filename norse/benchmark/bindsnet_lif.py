import time
import torch

# pytype: disable=import-error
from bindsnet.network import Network
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import LIFNodes, Input
from bindsnet.encoding import poisson
from bindsnet.network.monitors import Monitor

from benchmark import BenchmarkParameters

# pytype: enable=import-error


class BindsNetModule(torch.nn.Module):
    def __init__(self, parameters: BenchmarkParameters):
        super(BindsNetModule, self).__init__()
        network = Network(batch_size=parameters.batch_size, dt=parameters.dt)
        lif_nodes = LIFNodes(n=parameters.features)
        monitor = Monitor(
            obj=lif_nodes, state_vars=("s"), time=parameters.sequence_length
        )
        network.add_layer(Input(n=parameters.features), name="Input")
        network.add_layer(lif_nodes, name="Neurons")
        network.add_connection(
            Connection(
                source=network.layers["Input"], target=network.layers["Neurons"]
            ),
            source="Input",
            target="Neurons",
        )
        network.add_monitor(monitor, "Monitor")
        network.to(parameters.device)

        self.parameters = parameters
        self.network = network
        self.monitor = monitor

    def forward(self, input_data):
        self.network.run(
            inputs=input_data, time=self.parameters.dt * self.parameters.sequence_length
        )
        return self.monitor.get("s")


def lif_feed_forward_benchmark(parameters: BenchmarkParameters):
    with torch.no_grad():
        network = BindsNetModule(parameters)
        input_data = {
            "Input": poisson(
                datum=torch.rand(
                    (parameters.batch_size, parameters.features),
                    device=parameters.device,
                ).contiguous(),
                time=parameters.sequence_length,
                device=parameters.device,
            )
        }

        start = time.time()
        network(input_data)
        end = time.time()

        duration = end - start
        return duration
