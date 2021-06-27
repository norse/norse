import torch
from norse.torch import LIF
from norse.torch.utils.plot import plot_spikes_2d
plt.figure(figsize=(8, 4))
spikes, _ = LIF()(torch.randn(200, 10))
plot_spikes_2d(spikes)
plt.show()