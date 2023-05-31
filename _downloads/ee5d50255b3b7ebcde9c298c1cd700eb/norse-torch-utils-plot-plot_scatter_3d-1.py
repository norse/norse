import torch
from norse.torch.utils.plot import plot_scatter_3d
plt.figure(figsize=(10, 3))
distribution = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.02]))
data = distribution.sample(sample_shape=(3, 100, 10, 10)).squeeze()
data.names=('L', 'T', 'X', 'Y')
plot_scatter_3d(data)
plt.show()