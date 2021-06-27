import torch
from norse.torch.utils.plot import plot_heatmap_3d
data = torch.randn(4, 28, 28, names=('L', 'X', 'Y'))
plot_heatmap_3d(data)
plt.show()