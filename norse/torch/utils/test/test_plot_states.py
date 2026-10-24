import numpy as np
import pytest
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from norse.torch.functional.lif import LIFFeedForwardState
from norse.torch.utils.plot.plot import plot_neuron_states


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_plot_neuron_states_with_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    voltage = torch.tensor([0.1, 0.2, 0.4], device=device, requires_grad=True)
    state = LIFFeedForwardState(v=voltage, i=torch.zeros_like(voltage))
    figure = Figure()
    FigureCanvasAgg(figure)
    axes = figure.subplots()

    result = plot_neuron_states(state, "v", axes=axes)
    figure.canvas.draw()

    assert result is axes
    assert len(axes.lines) == 1
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [0.1, 0.2, 0.4])
    assert voltage.requires_grad
