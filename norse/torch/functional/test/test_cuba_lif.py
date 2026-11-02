import torch

from norse.torch.functional.cuba_lif import (
    CubaLIFFeedForwardState,
    CubaLIFParameters,
    cuba_lif_feed_forward_step,
)


def test_cuba_lif_feed_forward_step():
    p = CubaLIFParameters(
        tau_syn_inv=torch.as_tensor(100.0),
        tau_mem_inv=torch.as_tensor(100.0),
        r=torch.as_tensor(2.0),
        w_in=torch.as_tensor(0.5),
        v_th=torch.as_tensor(10.0),
    )
    s = CubaLIFFeedForwardState(v=torch.as_tensor(0.0), i=torch.as_tensor(0.0))
    xs = torch.tensor([1.0, 1.0, 0.0])

    # i_t = 0.9 * i_{t-1} + 0.5 * x_t
    # v_t = 0.9 * v_{t-1} + 0.1 * 2 * i_t
    expected_i = [0.5, 0.95, 0.855]
    expected_v = [0.1, 0.28, 0.423]

    for x, e_i, e_v in zip(xs, expected_i, expected_v):
        _, s = cuba_lif_feed_forward_step(x, s, p)
        assert torch.allclose(torch.as_tensor(e_i), s.i, atol=1e-5)
        assert torch.allclose(torch.as_tensor(e_v), s.v, atol=1e-5)


def test_cuba_lif_feed_forward_step_reset():
    p = CubaLIFParameters(v_th=torch.as_tensor(0.05), v_reset=torch.as_tensor(-1.0))
    s = CubaLIFFeedForwardState(v=torch.as_tensor(0.0), i=torch.as_tensor(0.0))
    z, s = cuba_lif_feed_forward_step(torch.as_tensor(1.0), s, p)
    assert z == 1.0
    assert s.v == -1.0


def test_cuba_lif_feed_forward_step_backward():
    x = torch.ones(10, requires_grad=True)
    s = CubaLIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10))
    for _ in range(10):
        z, s = cuba_lif_feed_forward_step(x, s)
    (z.sum() + s.v.sum()).backward()
    assert x.grad is not None
