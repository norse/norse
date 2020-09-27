import torch

from norse.torch.functional.stdp import (
    STDPState,
    linear_soft_multiplicative_stdp_step,
    linear_hard_multiplicative_stdp_step,
    linear_additive_stdp_step,
    conv2d_soft_multiplicative_stdp_step,
    conv2d_hard_multiplicative_stdp_step,
    conv2d_additive_stdp_step,
)


def test_linear_soft_multiplicative_stdp_step():
    time = 100
    pre, post = 2, 3
    w = torch.rand(post, pre)  # weights between 0 and 1 as default
    z_pre = torch.ones((time, 1, pre))
    z_post = (torch.randn(time, 1, post) > 0).float()
    s = STDPState(x=torch.zeros(1, pre), y=torch.zeros(1, post))

    for i in range(time):
        dw, s = linear_soft_multiplicative_stdp_step(w, z_pre[i], z_post[i], s)
        w += dw


def test_linear_hard_multiplicative_stdp_step():
    time = 100
    pre, post = 2, 3
    w = torch.rand(post, pre)  # weights between 0 and 1 as default
    z_pre = torch.ones((time, 1, pre))
    z_post = (torch.randn(time, 1, post) > 0).float()
    s = STDPState(x=torch.zeros(1, pre), y=torch.zeros(1, post))

    for i in range(time):
        dw, s = linear_hard_multiplicative_stdp_step(w, z_pre[i], z_post[i], s)
        w += dw


def test_linear_additive_stdp_step():
    time = 100
    pre, post = 2, 3
    w = torch.rand(post, pre)  # weights between 0 and 1 as default
    z_pre = torch.ones((time, 1, pre))
    z_post = (torch.randn(time, 1, post) > 0).float()
    s = STDPState(x=torch.zeros(1, pre), y=torch.zeros(1, post))

    for i in range(time):
        dw, s = linear_additive_stdp_step(w, z_pre[i], z_post[i], s)
        w += dw


def test_conv2d_soft_multiplicative_stdp_step():
    time = 100
    batch = 1
    in_channels, out_channels = 3, 2
    in_hw, out_hw = (10, 10), (7, 7)
    kernel = (4, 4)
    # stride, padding, dilation = (1, 0, 1) as default
    w = torch.nn.Conv2d(in_channels, out_channels, kernel).weight.clone()
    torch.nn.init.uniform_(w)  # weights between 0 and 1 as default
    z_pre = torch.ones((time, batch, in_channels, *in_hw))
    z_post = (torch.randn(time, batch, out_channels, *out_hw) > 0).float()
    s = STDPState(
        x=torch.zeros(batch, in_channels, *in_hw),
        y=torch.zeros(batch, out_channels, *out_hw),
    )

    for i in range(time):
        dw, s = conv2d_soft_multiplicative_stdp_step(w, z_pre[i], z_post[i], s)
        w += dw


def test_conv2d_hard_multiplicative_stdp_step():
    time = 1000
    batch = 1
    in_channels, out_channels = 3, 2
    in_hw, out_hw = (10, 10), (7, 7)
    kernel = (4, 4)
    # stride, padding, dilation = (1, 0, 1) as default
    w = torch.nn.Conv2d(in_channels, out_channels, kernel).weight.clone()
    torch.nn.init.uniform_(w)  # weights between 0 and 1 as default
    z_pre = torch.ones((time, batch, in_channels, *in_hw))
    z_post = (torch.randn(time, batch, out_channels, *out_hw) > 0).float()
    s = STDPState(
        x=torch.zeros(batch, in_channels, *in_hw),
        y=torch.zeros(batch, out_channels, *out_hw),
    )

    for i in range(time):
        dw, s = conv2d_hard_multiplicative_stdp_step(w, z_pre[i], z_post[i], s)
        w += dw


def test_conv2d_additive_stdp_step():
    time = 100
    batch = 1
    in_channels, out_channels = 3, 2
    in_hw, out_hw = (10, 10), (7, 7)
    kernel = (4, 4)
    # stride, padding, dilation = (1, 0, 1) as default
    w = torch.nn.Conv2d(in_channels, out_channels, kernel).weight.clone()
    torch.nn.init.uniform_(w)  # weights between 0 and 1 as default
    z_pre = torch.ones((time, batch, in_channels, *in_hw))
    z_post = (torch.randn(time, batch, out_channels, *out_hw) > 0).float()
    s = STDPState(
        x=torch.zeros(batch, in_channels, *in_hw),
        y=torch.zeros(batch, out_channels, *out_hw),
    )

    for i in range(time):
        dw, s = conv2d_additive_stdp_step(w, z_pre[i], z_post[i], s)
        w += dw
