import torch


def heaviside(data):
    """
    A `heaviside step function <https://en.wikipedia.org/wiki/Heaviside_step_function>`_
    that truncates numbers <= 0 to 0 and everything else to 1.

    .. math::
        H[n]=\\begin{cases} 0, & n <= 0, \\ 1, & n \g 0, \end{cases}
    """
    return torch.where(
        data <= torch.zeros_like(data),
        torch.zeros_like(data),
        torch.ones_like(data),
    )
