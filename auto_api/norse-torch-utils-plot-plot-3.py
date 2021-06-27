import torch
from norse.torch import LIF
from norse.torch.utils.plot import plot_histogram_2d
spikes, state = LIF()(torch.ones(10, 10) + torch.randn(10, 10))
plot_histogram_2d(state.v)
plt.show()