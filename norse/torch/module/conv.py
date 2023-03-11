import torch
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair


# pytype: disable=module-attr
class LConv2d(torch.nn.Conv3d):
    # pytype: enable=module-attr
    """Implements a 2d-convolution applied pointwise in time.
    See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d,
    for documentation of the arguments, which we will reproduce in part here.

    This module expects an additional temporal dimension in the tensor it is passed, that is
    in the notation in the documentation referenced above, it turns in the simplest case a tensor
    with input shape :math:`(T, N, C_{\text{in}}, H, W)` and output tensor of shape
    :math:`(T, N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`, by applying a 2d convolution
    operation pointwise along the time-direction, with T denoting the number of time steps.

    {groups_note}

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:
        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
            and the second `int` for the width dimension

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        device=None,
        dtype=None,
    ):
        kernel_size = (*_pair(kernel_size), 1)
        stride = (*_pair(stride), 1)
        padding = (*_pair(padding), 0)
        dilation = (*_pair(dilation), 1)

        super(LConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=False,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )

    def forward(self, input_tensor: torch.Tensor):
        return F.conv3d(
            input_tensor.permute(1, 2, 3, 4, 0),
            self.weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        ).permute(
            4, 0, 1, 2, 3
        )  # TODO: how much of a performance impact do the permutations have
