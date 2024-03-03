"""
Tensorboard related utilities.
"""

from typing import Any, Callable, Optional, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter


def __generate_hook(
    f: Callable[[torch.nn.Module, torch.Tensor, Any], Any],
    label: str,
    counter_name: str,
    aggregation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    def hook(mod, _, out):
        if not hasattr(mod, counter_name):
            setattr(mod, counter_name, 0)
        out = out[0] if aggregation is None else aggregation(out[0])
        f(label, out, getattr(mod, counter_name))
        setattr(mod, counter_name, getattr(mod, counter_name) + 1)

    return hook


def hook_spike_activity_mean(
    key: str, writer: SummaryWriter
) -> Callable[[torch.nn.Module, Tuple[torch.Tensor, Any], Any], None]:
    """
    Generates a hook that can be applied to stateful torch Modules.
    The hook plots the *mean neuron activity* as a line, assuming that the module output is a
    tuple of (spikes, state). That is, this will not work on modules that only
    returns a single tensor as output (i. e. have no state).

    Read more about hooks in PyTorch Modules in the
    `Module documentation <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_.

    Example:
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> import norse.torch as snn
        >>> from snn.util import tensorboard
        >>> hook = tensorboard.hook_spike_activity_mean("lif", SummaryWriter())
        >>> snn.LIFCell().register_forward_hook(hook)

    Arguments:
        key (str): The name of the module as referred to in Tensorboard
        writer (SummaryWriter): The Tensorboard Writer to log to.

    Returns:
        A hook that will plot the mean neuron spike activity when registered
        to a Torch module.
    """
    return __generate_hook(
        writer.add_image, key, "norse_hook_spike_activity_mean", aggregation=torch.mean
    )


def hook_spike_activity_sum(
    key: str, writer: SummaryWriter
) -> Callable[[torch.nn.Module, Tuple[torch.Tensor, Any], Any], None]:
    """
    Generates a hook that can be applied to stateful torch Modules.
    The hook plots the *summed neuron activity* as a line, assuming that the module output is a
    tuple of (spikes, state). That is, this will not work on modules that only
    returns a single tensor as output (i. e. have no state).

    Read more about hooks in PyTorch Modules in the
    `Module documentation <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_.

    Example:
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> import norse.torch as snn
        >>> from snn.util import tensorboard
        >>> hook = tensorboard.hook_spike_activity_sum("lif", SummaryWriter())
        >>> snn.LIFCell().register_forward_hook(hook)

    Arguments:
        key (str): The name of the module as referred to in Tensorboard
        writer (SummaryWriter): The Tensorboard Writer to log to.

    Returns:
        A hook that will plot the mean neuron spike activity when registered
        to a Torch module.
    """
    return __generate_hook(
        writer.add_scalar, key, "norse_hook_spike_activity_sum", aggregation=torch.sum
    )


def hook_spike_histogram_mean(
    key: str, writer: SummaryWriter
) -> Callable[[torch.nn.Module, Tuple[torch.Tensor, Any], Any], None]:
    """
    Generates a hook that can be applied to stateful torch Modules.
    The hook shows a histogram of *mean neuron activity*, assuming that the module output is a
    tuple of (spikes, state). That is, this will not work on modules that only
    returns a single tensor as output (i. e. have no state).

    Read more about hooks in PyTorch Modules in the
    `Module documentation <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_.

    Example:
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> import norse.torch as snn
        >>> from snn.util import tensorboard
        >>> hook = tensorboard.hook_spike_histogram_mean("lif", SummaryWriter())
        >>> snn.LIFCell().register_forward_hook(hook)

    Arguments:
        key (str): The name of the module as referred to in Tensorboard
        writer (SummaryWriter): The Tensorboard Writer to log to.

    Returns:
        A hook that will plot the mean neuron spike activity when registered
        to a Torch module.
    """
    return __generate_hook(
        writer.add_histogram,
        key,
        "norse_hook_spike_activity_mean",
        aggregation=torch.mean,
    )


def hook_spike_histogram_sum(
    key: str, writer: SummaryWriter
) -> Callable[[torch.nn.Module, Tuple[torch.Tensor, Any], Any], None]:
    """
    Generates a hook that can be applied to stateful torch Modules.
    The hook shows a histogram of the *summed neuron activity*, assuming that the module
    output is a tuple of (spikes, state). That is, this will not work on modules that only
    returns a single tensor as output (i. e. have no state).

    Read more about hooks in PyTorch Modules in the
    `Module documentation <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_.

    Example:
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> import norse.torch as snn
        >>> from snn.util import tensorboard
        >>> hook = tensorboard.hook_spike_histogram_sum("lif", SummaryWriter())
        >>> snn.LIFCell().register_forward_hook(hook)

    Arguments:
        key (str): The name of the module as referred to in Tensorboard
        writer (SummaryWriter): The Tensorboard Writer to log to.

    Returns:
        A hook that will plot the mean neuron spike activity when registered
        to a Torch module.
    """
    return __generate_hook(
        writer.add_histogram,
        key,
        "norse_hook_spike_activity_mean",
        aggregation=torch.sum,
    )


def hook_spike_image(
    key: str, writer: SummaryWriter
) -> Callable[[torch.nn.Module, Tuple[torch.Tensor, Any], Any], None]:
    """
    Generates a hook that can be applied to stateful torch Modules.
    The hook plots the *spiked* output, assuming that the module output is a
    tuple of (spikes, state). That is, this will not work on modules that only
    returns a single tensor as output (i. e. have no state).

    Read more about hooks in PyTorch Modules in the
    `Module documentation <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_.

    Example:
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> import norse.torch as snn
        >>> from snn.util import tensorboard
        >>> hook = tensorboard.hook_spike_image("lif", SummaryWriter())
        >>> snn.LIFCell().register_forward_hook(hook)

    Arguments:
        key (str): The name of the module as referred to in Tensorboard
        writer (SummaryWriter): The Tensorboard Writer to log to.

    Returns:
        A hook that will plot the spike activity as an image when registered
        to a Torch module.
    """
    return __generate_hook(writer.add_image, key, "norse_hook_spike_image")
