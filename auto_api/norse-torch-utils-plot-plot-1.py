import torch
from norse.torch.utils.plot import plot_heatmap_2d
data = torch.randn(28, 28)
plot_heatmap_2d(data)
plt.show()