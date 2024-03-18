import pytest
import platform

import torch
from norse.torch.functional.lif_box import LIFBoxFeedForwardState, LIFBoxParameters
from norse.torch.module.lif_box import LIFBoxCell


def test_lif_box_cell_feed_forward_step_batch():
    x = torch.ones(2, 1)
    s = LIFBoxFeedForwardState(v=torch.zeros(2, 1))

    z, s = LIFBoxCell()(x, s)
    assert z.shape == (2, 1)
    assert torch.all(torch.eq(s.v, 0.1))


def test_lif_box_cell_backward():
    x = torch.ones(2, 1)

    z, s = LIFBoxCell()(x)
    z.sum().backward()
    assert s.v.grad_fn is not None


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
def test_lif_box_compile_cpu():
    x = torch.ones(2, 1)
    p = LIFBoxParameters(
        tau_mem_inv=torch.ones(1) * 1000,
        v_th=torch.ones(1),
        v_leak=torch.zeros(1),
        v_reset=torch.zeros(1),
        alpha=torch.zeros(1),
    )

    m = LIFBoxCell(p)
    m = torch.compile(m)
    z, s = m(x)
    _, s = m(x, s)

    z.sum().backward()
    assert s.v.grad_fn is not None
    assert z.shape == (2, 1)
    assert torch.all(torch.eq(s.v, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
def test_lif_box_compile_gpu():
    x = torch.ones(2, 1, device="cuda")
    p = LIFBoxParameters(
        tau_mem_inv=torch.ones(1, device="cuda") * 1000,
        v_th=torch.ones(1, device="cuda"),
        v_leak=torch.zeros(1, device="cuda"),
        v_reset=torch.zeros(1, device="cuda"),
        alpha=torch.zeros(1, device="cuda"),
    )

    m = LIFBoxCell(p)
    m = torch.compile(m, mode="reduce-overhead")
    z, s = m(x)
    _, s = m(x, s)

    z.sum().backward()
    assert s.v.grad_fn is not None
    assert z.shape == (2, 1)
    assert torch.all(torch.eq(s.v, 1))
