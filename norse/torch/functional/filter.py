r"""
An `Exponential smoothing or exponential filter <https://wiki2.org/en/Exponential_smoothing>`_
that smoothing time series data using the exponential window function.

.. math::
   s(t) = x(y) + \alpha * s(t - \Delta t),
where smoothing factor
.. math::
    \alpha = e^{-\Delta t * \tau_{filter_inv}}.
"""

import torch


@torch.jit.script
def _exp_filter_step_jit(
    old_value: torch.Tensor, new_value: torch.Tensor, parameter: float
) -> torch.Tensor:
    value_new = parameter * old_value + new_value
    return value_new


def exp_filter_step(
    old_value: torch.Tensor, new_value: torch.Tensor, parameter: float
) -> torch.Tensor:
    return _exp_filter_step_jit(old_value, new_value, parameter)
