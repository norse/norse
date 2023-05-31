import torch
from norse.torch import LIF
from norse.torch.utils.plot import plot_neuron_states
data = torch.ones(10, 3) * torch.tensor([0.0, 0.1, 0.3])
_, states = LIF(record_states=True)(data)
plot_neuron_states(states, "i")