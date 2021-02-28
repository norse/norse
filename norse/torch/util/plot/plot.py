"""
Utilities for plotting spikes in layers over time in 2D and 3D.
"""

from typing import NamedTuple, Optional
import matplotlib.pyplot as plt

import torch

_pixels_to_inches = lambda x: x * 1e-2


def _init_axis(ax: plt.Axes, p: NamedTuple):
    ax.set_title(p.title)
    ax.set_xlabel(p.x_label)
    ax.set_ylabel(p.y_label)
    ax.set_zlabel(p.z_label)


class PlotParameters(NamedTuple):
    """
    Parameters for plotting in Matplotlib.

    Arguments:
        title (str): Name of the plot. Defaults to None
        x_label (str): X label of the plot. Defaults to None
        y_label (str): Y label (in the horizontal plane). Defaults to None
        z_label (str): Z label (in the vertical plane). Defaults to None
        c_label (str): Label for the color bar. Defaults to None
        show_colorbar (bool): Whether to display a color bar or not, if applicable. Defaults to True
        width (int): Width of the entire plot **in pixels**
        height (int): Height of the entire plot **in pixels**

    """

    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    z_label: Optional[str] = None
    c_label: Optional[str] = None
    show_colorbar: bool = True
    width: int = 640
    height: int = 480


def plot_heatmap_2d(
    spikes: torch.Tensor, p: PlotParameters = PlotParameters(), **kwargs
):
    """
    Plots a heatmat of spikes from a single layer

    Example:
        >>> data = torch.randn(28, 28)
        >>> plot_histogram_2d(data)

    Arguments:
        spikes (torch.Tensor): A tensor of spikes from a single layer in two dimensions.
        p (PlotParameters): Configuration for the overall matplotlib figure.
        kwargs: Specific key-value arguments to style the figure
                fed to the :meth:`matplotlib.pyplot.imshow`_ function.
    """
    figure = plt.figure(
        figsize=(_pixels_to_inches(p.width), _pixels_to_inches(p.height))
    )
    pos = plt.imshow(spikes, interpolation="none", aspect="auto", **kwargs)
    if p.show_colorbar:
        figure.colorbar(pos, ax=plt.gca(), label=p.c_label)
    return figure


def plot_histogram_2d(
    spikes: torch.Tensor, p: PlotParameters = PlotParameters(), **kwargs
):
    """
    Plots a histogram of layer activity.

    Example:
        >>> data = torch.randn(28, 28).flatten()
        >>> plot_histogram_2d(data)

    Arguments:
        spikes (torch.Tensor): A tensor of spikes from a single layer in one dimension.
        p (PlotParameters): Configuration for the overall matplotlib figure.
        kwargs: Specific key-value arguments to style the figure
                fed to the :meth:`matplotlib.pyplot.hist`_ function.
    """
    figure = plt.figure(
        figsize=(_pixels_to_inches(p.width), _pixels_to_inches(p.height))
    )
    plt.hist(spikes, bins=p.bins, density=True, **kwargs)
    return figure


def plot_scatter_3d(
    spikes: torch.Tensor, p: PlotParameters = PlotParameters(), **kwargs
):
    """
    Plots spike activity in time. If multiple layers are given, the the layers will be
    shown in subplots.
    Expects a named tensor.

    Example:
        >>> data = torch.randn(3, 100, 28, 28, names=('L', 'T', 'X', 'Y'))
        >>> plot_scatter_3d(data)

    Arguments:
        spikes (torch.NamedTensor): A tensor named with four dimensions: T (time), L (layer), X, Y.
                                    Expected to be in the range :math:`[0, 1]`.
        p (PlotParameters): Configuration for the overall matplotlib figure.
        kwargs: Specific key-value arguments to style the figure
                fed to the :meth:`matplotlib.pyplot.scatter` function.
    """
    fig = plt.figure(figsize=(_pixels_to_inches(p.width), _pixels_to_inches(p.height)))
    spikes = spikes.align_to(*"TLXY")
    T = spikes.shape[0]
    L = spikes.shape[1]

    axes = []
    for l in range(L):
        ax = fig.add_subplot(1, L, l + 1, projection="3d")
        axes.append(ax)
        _init_axis(ax, p)
        unnamed = spikes[:, l]
        unnamed.names = None  # Unnamed tensor required for to_sparse
        s = unnamed.to_sparse().coalesce()

        ax.invert_yaxis()
        ax.invert_zaxis()
        ax.set_xlim([0, T])
        pos = ax.scatter(
            s.indices()[0], s.indices()[2], s.indices()[1], c=s.values(), **kwargs
        )
    if p.show_colorbar:
        fig.colorbar(pos, ax=axes, label=p.c_label, fraction=0.06 / len(axes))
    return fig


def plot_heatmap_3d(
    spikes: torch.Tensor, p: PlotParameters = PlotParameters(), **kwargs
):
    """
    Plots heatmaps for several layers in time.
    Expects a named tensor.
    Instead of using the :meth:`matplotlib.pyplot.imshow`_ matplotlib function,
    we make use of the :meth:`matplotlib.pyplot.scatter` function to disperse points in 3d.

    Example:
        >>> data = torch.randn(3, 100, 28, 28, names=('L', 'X', 'Y'))
        >>> plot_heatmap_3d(data)

    Arguments:
        spikes (torch.NamedTensor): A tensor named with four dimensions: T (time), L (layer), X, Y.
                                    Expected to be in the range :math:`[0, 1]`.
        p (PlotParameters): Configuration for the overall matplotlib figure.
        kwargs: Specific key-value arguments to style the figure
                fed to the :meth:`matplotlib.pyplot.scatter`_ function.
    """
    fig = plt.figure(figsize=(_pixels_to_inches(p.width), _pixels_to_inches(p.height)))
    spikes = spikes.align_to(*"LXY")
    L = spikes.shape[0]

    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _init_axis(ax, p)
    unnamed = spikes + 1e-10  # Add infinitely small amount to "trigger" all pixels
    unnamed.names = None  # Unnamed tensor required for to_sparse
    s = unnamed.to_sparse().coalesce()

    if "c" not in kwargs:
        kwargs["c"] = s.values()

    ax.set_title(None)
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.set_xlim([0, L - 1])
    ax.set_ylim([0, s.shape[2]])
    ax.set_zlim([0, s.shape[1]])
    plt.xticks(range(0, L), range(1, L + 1))
    pos = ax.scatter(s.indices()[0], s.indices()[2], s.indices()[1], **kwargs)
    if p.show_colorbar:
        fig.colorbar(pos, ax=ax, label=p.c_label)
    return fig
