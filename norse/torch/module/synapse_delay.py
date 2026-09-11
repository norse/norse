r"""
See :mod:`norse.torch.functional.synapse_delay` for the underlying step
function and the motivation (issue #224).
"""
import math
from typing import Optional, Union

import torch

from norse.torch.functional.synapse_delay import (
    SynapseDelayState,
    synapse_delay_step,
)


class SynapseDelay(torch.nn.Module):
    r"""
    Delays a signal by ``delay`` time steps before passing it on -- an
    axonal / synaptic transmission delay. Drop it between two neuron
    layers in a :class:`SequentialState`::

        model = SequentialState(
            LIFCell(),
            SynapseDelay(delay=5.0, max_delay=16, learn_delay=True),
            LIFCell(),
        )

    The delay need not be an integer number of steps: it is realised by
    interpolation over a ring buffer of the ``max_delay`` most recent
    inputs, so ``delay`` is a continuous, differentiable quantity when
    ``learn_delay=True``.

    Two interpolation kernels are available. ``kernel="linear"`` (default)
    is an exact 2-tap kernel with a *local* gradient (about +-1 step) --
    good for fine delays or fine-tuning from a nearby initial value.
    ``kernel="gaussian"`` gives gradient across the whole delay range;
    anneal ``sigma`` from wide to narrow over training to learn a large
    delay starting from a poor initial guess (Hammouamri,
    Khalfaoui-Hassani & Masquelier, *Learning Delays in Spiking Neural
    Networks using Dilated Convolutions with Learnable Spacings*, ICLR
    2024).

    ``delay`` may be a scalar (one delay shared by every element) or a
    tensor broadcastable against the module's input (e.g. one delay per
    feature channel for a ``(batch, channels)`` input).

    Parameters:
        delay (Union[float, torch.Tensor]): initial delay, in time steps.
            Defaults to 1.0.
        max_delay (Optional[int]): length of the ring buffer / kernel. Must
            be >= ``ceil(delay)``. Defaults to ``ceil(delay) + 1``.
        learn_delay (bool): if True, ``delay`` is a learnable
            ``torch.nn.Parameter``; otherwise it is fixed. Defaults to
            False.
        kernel (str): ``"linear"`` (default) or ``"gaussian"``.
        sigma (float): width of the Gaussian kernel (ignored for
            ``"linear"``). Defaults to 1.0.

    Examples:
        >>> import torch
        >>> from norse.torch import SynapseDelay
        >>> delay = SynapseDelay(delay=3.0, max_delay=8)
        >>> x = torch.zeros(1, 4)
        >>> x[0, 0] = 1.0
        >>> out, state = delay(x)  # a unit impulse; appears 3 steps later
    """

    def __init__(
        self,
        delay: Union[float, torch.Tensor] = 1.0,
        max_delay: Optional[int] = None,
        learn_delay: bool = False,
        kernel: str = "linear",
        sigma: float = 1.0,
    ):
        super().__init__()
        if not isinstance(delay, torch.Tensor):
            delay = torch.as_tensor(delay, dtype=torch.float32)
        delay = delay.float()

        if max_delay is None:
            max_delay = max(1, int(math.ceil(float(delay.max()))) + 1)
        if int(max_delay) != max_delay or max_delay < 1:
            raise ValueError("max_delay must be a positive integer.")
        max_delay = int(max_delay)

        if float(delay.max()) > max_delay or float(delay.min()) < 0:
            raise ValueError(
                "every delay must lie in [0, max_delay] "
                f"(got range [{float(delay.min())}, {float(delay.max())}], "
                f"max_delay={max_delay})."
            )
        if kernel not in ("linear", "gaussian"):
            raise ValueError("kernel must be 'linear' or 'gaussian'.")

        self.max_delay = max_delay
        self.kernel = kernel
        self.sigma = float(sigma)
        self.learn_delay = learn_delay
        if learn_delay:
            self.delay = torch.nn.Parameter(delay)
        else:
            self.register_buffer("delay", delay)

    def forward(
        self,
        input_tensor: torch.Tensor,
        state: Optional[SynapseDelayState] = None,
    ):
        return synapse_delay_step(
            input_tensor,
            state,
            self.delay,
            self.max_delay,
            kernel=self.kernel,
            sigma=self.sigma,
        )

    def extra_repr(self) -> str:
        d = self.delay.detach()
        dstr = f"{float(d):.3g}" if d.dim() == 0 else f"tensor{tuple(d.shape)}"
        return (
            f"delay={dstr}, max_delay={self.max_delay}, "
            f"learn_delay={self.learn_delay}, kernel={self.kernel}"
            + (f", sigma={self.sigma}" if self.kernel == "gaussian" else "")
        )
