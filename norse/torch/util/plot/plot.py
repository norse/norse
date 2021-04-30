"""
Utilities for plotting spikes in layers over time in 2D and 3D.
"""

from typing import List

import matplotlib.pyplot as plt

import torch


def _detach_tensor(tensor: torch.Tensor):
    if tensor.requires_grad:
        return tensor.detach()
    else:
        return tensor


def plot_heatmap_2d(
    data: torch.Tensor, axes: plt.Axes = None, show_colorbar: bool = False, **kwargs
):
    """
    Plots a heatmat of two-dimensional data

    Example:
        >>> data = torch.randn(28, 28)
        >>> plot_heatmap_2d(data)

    .. plot::

        import torch
        from norse.torch import plot_heatmap_2d
        data = torch.randn(28, 28)
        plot_heatmap_2d(data)
        plt.show()

    Arguments:
        data (torch.Tensor): A tensor of data to plot in two dimensions
                             (typically spikes with time in the first dimension and
                             neuron id in the second).
        axes (matplotlib.axes.Axes): The matplotlib axis to plot on, if any.
                                     Defaults to :meth:`matplotlib.pyplot.gca`
        show_colorbar (bool): Show a colorbar (True) or not (False).
        kwargs: Specific key-value arguments to style the figure
                fed to the :meth:`matplotlib.pyplot.imshow` function.

    Returns:
        A :class:`matplotlib.axes.Axes`.
    """
    ax = axes if axes is not None else plt.gca()
    kwargs["aspect"] = kwargs.get("aspect", "auto")
    kwargs["interpolation"] = kwargs.get("interpolation", "none")

    pos = plt.imshow(_detach_tensor(data).T, **kwargs)
    if show_colorbar:
        ax.colorbar(pos, ax=ax)
    return ax


def plot_histogram_2d(data: torch.Tensor, axes: plt.Axes = None, **kwargs):
    """
    Plots a histogram of 1-dimensional data.

    Example:
        >>> data = torch.randn(28, 28).flatten()
        >>> plot_histogram_2d(data)

    .. plot::

        import torch
        from norse.torch import LIF, plot_histogram_2d
        spikes, state = LIF()(torch.ones(100, 1))
        plot_histogram_2d(spikes)
        plt.show()

    Arguments:plot_scatter_3d(data.mean(1))
        data (torch.Tensor): A tensor of single-dimensional data.
        axes (matplotlib.axes.Axes): The matplotlib axis to plot on, if any.
                                     Defaults to :meth:`matplotlib.pyplot.gca`
        kwargs: Specific key-value arguments to style the figure
                fed to the :meth:`matplotlib.pyplot.hist` function.

    Returns:
        An :class:`matplotlib.axes.Axes`.
    """
    ax = axes if axes is not None else plt.gca()
    kwargs["density"] = kwargs.get("density", True)
    plt.hist(_detach_tensor(data).numpy(), **kwargs)
    return ax


def plot_spikes_2d(spikes: torch.Tensor, axes: plt.Axes = None, **kwargs):
    """
    Plots a 2D diagram of spikes. Works similar to the :meth:`plot_heatmap_2d` but
    in black and white.

    Arguments:
        spikes (torch.Tensor): A tensor of spikes from a single layer in two dimensions.
        axes (matplotlib.axes.Axes): The matplotlib axis to plot on, if any.
                                     Defaults to :meth:`matplotlib.pyplot.gca`
        kwargs: Specific key-value arguments to style the figure
                fed to the :meth:`matplotlib.pyplot.imshow` function.

    Example:
        >>> import torch
        >>> from norse.torch import LIF, plot_spikes_2d
        >>> spikes, _ = LIF()(torch.randn(200, 10))
        >>> plot_spikes_2d(spikes)
        >>> plt.show()

    .. plot::

        import torch
        from norse.torch import LIF, plot_spikes_2d
        spikes, _ = LIF()(torch.randn(200, 10))
        plot_spikes_2d(spikes)
        plt.show()

    Returns:
        An :class:`matplotlib.axes.Axes` object
    """
    kwargs["cmap"] = kwargs.get("cmap", "binary")
    return plot_heatmap_2d(spikes, axes=axes, **kwargs)


def plot_scatter_3d(
    spikes: torch.Tensor,
    axes: List[plt.Axes] = None,
    show_colorbar: bool = True,
    **kwargs
):
    """
    Plots spike activity in time. If multiple layers are given, the layers will be
    shown in subplots.
    Expects a named tensor in three dimensions (L, X, Y) or four, with time (T, L, X, Y).

    Example:
        >>> distribution = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.02]))
        >>> data = distribution.sample(sample_shape=(3, 100, 10, 10)).squeeze()
        >>> data.names = ('L', 'T', 'X', 'Y')
        >>> plot_scatter_3d(data)

    .. plot::

        import torch
        from norse.torch import LIF, plot_scatter_3d
        distribution = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.02]))
        data = distribution.sample(sample_shape=(3, 100, 10, 10)).squeeze()
        data.names=('L', 'T', 'X', 'Y')
        plot_scatter_3d(data)
        plt.show()

    Arguments:
        spikes (torch.NamedTensor): A tensor named with four dimensions: T (time), L (layer), X, Y.
                                    Expected to be in the range :math:`[0, 1]`.
        axes (List[plt.Axes]): A list of Axes that should have the same length as the L dimension in
                               the spike tensor. Defaults to None, which will generate a grid for you.
        show_colorbar (bool): Show a colorbar (True) or not (False).
        kwargs: Specific key-value arguments to style the figure
                fed to the :meth:`matplotlib.pyplot.scatter` function.

    Returns:
        A list of :class:`matplotlib.axes.Axes`
    """
    if len(spikes.shape) > 3:
        spikes = _detach_tensor(spikes).align_to(*"TLXY")
    else:
        spikes = _detach_tensor(spikes).align_to(*"LXY").unsqueeze(0)
    T = spikes.shape[0]
    L = spikes.shape[1]

    axes = [None] * L if axes is None else axes
    for l, axis in enumerate(axes):
        if axis is not None:
            ax = axis
        else:
            ax = plt.gcf().add_subplot(1, L, l + 1, projection="3d")
            axes[l] = ax

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
        fig.colorbar(pos, ax=axes, fraction=0.06 / len(axes))
    return fig


def plot_heatmap_3d(spikes: torch.Tensor, show_colorbar: bool = False, **kwargs):
    """
    Plots heatmaps for some activity in several layers.
    Expects a named tensor with names=('L', 'X', 'Y').
    Instead of using the :meth:`matplotlib.pyplot.imshow` matplotlib function,
    we make use of the :meth:`matplotlib.pyplot.scatter` function to disperse points in 3d.

    Example:
        >>> import torch
        >>> from norse.torch import plot_heatmap_3d
        >>> data = torch.randn(4, 28, 28, names=('L', 'X', 'Y'))
        >>> plot_heatmap_3d(data)
        >>> plt.show()

    .. plot::

        import torch
        from norse.torch import plot_heatmap_3d
        data = torch.randn(4, 28, 28, names=('L', 'X', 'Y'))
        plot_heatmap_3d(data)
        plt.show()

    Arguments:
        spikes (torch.NamedTensor): A tensor named with four dimensions: T (time), L (layer), X, Y.
                                    Expected to be in the range :math:`[0, 1]`.
        show_colorbar (bool): Show a colorbar (True) or not (False).
        kwargs: Specific key-value arguments to style the figure
                fed to the :meth:`matplotlib.pyplot.scatter` function.

    Returns:
        An :class:`matplotlib.axes.Axes` object
    """
    spikes = _detach_tensor(spikes).align_to(*"LXY")
    L = spikes.shape[0]

    ax = plt.gcf().add_subplot(1, 1, 1, projection="3d")
    unnamed = spikes + 1e-10  # Add infinitely small amount to "trigger" all pixels
    unnamed.names = None  # Unnamed tensor required for to_sparse
    s = unnamed.to_sparse().coalesce()

    kwargs["c"] = kwargs.get("c", s.values())

    ax.set_title(None)
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.set_xlim([0, L - 1])
    ax.set_ylim([0, s.shape[2]])
    ax.set_zlim([0, s.shape[1]])
    plt.xticks(range(0, L), range(1, L + 1))
    pos = ax.scatter(s.indices()[0], s.indices()[2], s.indices()[1], **kwargs)
    if show_colorbar:
        plt.gcf().colorbar(pos, ax=ax, label=p.c_label)
    return ax
