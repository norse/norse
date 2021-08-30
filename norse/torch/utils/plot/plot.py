"""
Utilities for plotting spikes in layers over time in 2D and 3D.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from norse.torch.functional.izhikevich import (
    IzhikevichSpikingBehavior,
    izhikevich_feed_forward_step,
)


def _detach_tensor(tensor: torch.Tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu()
    else:
        return tensor.cpu()


def plot_heatmap_2d(
    data: torch.Tensor,
    axes: Optional[plt.Axes] = None,
    show_colorbar: bool = False,
    **kwargs
):
    """
    Plots a heatmat of two-dimensional data

    Example:
        >>> data = torch.randn(28, 28)
        >>> plot_heatmap_2d(data)

    .. plot::

        import torch
        from norse.torch.utils.plot import plot_heatmap_2d
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


def plot_heatmap_3d(spikes: torch.Tensor, show_colorbar: bool = False, **kwargs):
    """
    Plots heatmaps for some activity in several layers.
    Expects a named tensor with names=('L', 'X', 'Y').
    Instead of using the :meth:`matplotlib.pyplot.imshow` matplotlib function,
    we make use of the :meth:`matplotlib.pyplot.scatter` function to disperse points in 3d.

    Example:
        >>> import torch
        >>> from norse.torch.utils.plot import plot_heatmap_3d
        >>> data = torch.randn(4, 28, 28, names=('L', 'X', 'Y'))
        >>> plot_heatmap_3d(data)
        >>> plt.show()

    .. plot::

        import torch
        from norse.torch.utils.plot import plot_heatmap_3d
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
        plt.gcf().colorbar(pos, ax=ax)
    return ax


def plot_histogram_2d(spikes: torch.Tensor, axes: Optional[plt.Axes] = None, **kwargs):
    """
    Plots a histogram of 1-dimensional data.

    Example:
        >>> cell = LIF()
        >>> data = torch.ones(10, 10) + torch.randn(10, 10)
        >>> spikes, state = cell(data)
        >>> plot_histogram_2d(state.v)

    .. plot::

        import torch
        from norse.torch import LIF
        from norse.torch.utils.plot import plot_histogram_2d
        spikes, state = LIF()(torch.ones(10, 10) + torch.randn(10, 10))
        plot_histogram_2d(state.v)
        plt.show()


    Arguments:
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
    plt.hist(_detach_tensor(spikes).numpy(), **kwargs)
    return ax


def plot_scatter_3d(
    spikes: torch.Tensor,
    axes: Optional[List[plt.Axes]] = None,
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
        from norse.torch.utils.plot import plot_scatter_3d
        plt.figure(figsize=(10, 3))
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
    if show_colorbar:
        plt.gcf().colorbar(pos, ax=axes, fraction=0.04 / len(axes))
    return axes


def plot_spikes_2d(spikes: torch.Tensor, axes: plt.Axes = None, **kwargs):
    """
    Plots a 2D diagram of spikes. Works similar to the :meth:`plot_heatmap_2d` but
    in black and white.

    Example:
        >>> import torch
        >>> from norse.torch import LIF
        >>> from norse.torch.utils.plot import plot_spikes_2d
        >>> spikes, _ = LIF()(torch.randn(200, 10))
        >>> plot_spikes_2d(spikes)
        >>> plt.show()

    .. plot::

        import torch
        from norse.torch import LIF
        from norse.torch.utils.plot import plot_spikes_2d
        plt.figure(figsize=(8, 4))
        spikes, _ = LIF()(torch.randn(200, 10))
        plot_spikes_2d(spikes)
        plt.show()


    Arguments:
        spikes (torch.Tensor): A tensor of spikes from a single layer in two dimensions.
        axes (matplotlib.axes.Axes): The matplotlib axis to plot on, if any.
                                     Defaults to :meth:`matplotlib.pyplot.gca`
        kwargs: Specific key-value arguments to style the figure
                fed to the :meth:`matplotlib.pyplot.imshow` function.

    Returns:
        An :class:`matplotlib.axes.Axes` object
    """
    kwargs["cmap"] = kwargs.get("cmap", "binary")
    return plot_heatmap_2d(spikes, axes=axes, **kwargs)


def plot_izhikevich(
    behavior: IzhikevichSpikingBehavior,
    current: float = 1,
    time_print: int = 250,
    time_current: int = 20,
    timestep_print: float = 0.1,
):
    """
    Computes and plots a 2D visualisation of the behavior of an Izhikevich neuron model.
    By default, the time window is 250ms with a time step of 0.1ms

    Example :
        >>> import torch
        >>> from norse.torch import tonic_spiking
        >>> from norse.torch.utils.plot import plot_izhikevich
        >>> plot_izhikevich(tonic_spiking)

    .. plot::

        import torch
        from norse.torch import tonic_spiking
        from norse.torch.utils.plot import plot_izhikevich
        plot_izhikevich(tonic_spiking)

    Arguments :
        behavior (IzhikevichSpikingBehavior) : behavior of an Izhikevich neuron
        current (float) : strengh of the input current, defaults to 1
        time_print (float) : size of the time window for simulation (in ms)
        time_current (float) : time at which the input current goes from 0 to current (in ms)
        timestep_print (float) : timestep of the simulation (in ms)
    """
    p, s = behavior
    T1 = time_current
    vs = []
    us = []
    cs = []
    time = []

    for t in np.arange(0, time_print, timestep_print):
        vs.append(s.v.item())
        us.append(s.u.item())
        time.append(t * timestep_print)

        if t > T1:
            input_current = current * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)
        _, s = izhikevich_feed_forward_step(input_current, s, p)
        cs.append(input_current)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel("Membrane potential (mV)")
    ax1.set_xlabel("Time (ms)")
    ax1.plot(time, vs)
    ax1.plot(time, cs)
