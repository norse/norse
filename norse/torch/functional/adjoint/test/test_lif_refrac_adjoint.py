import torch
import numpy as np

from norse.torch.functional.adjoint.lif_refrac_adjoint import (
    LIFFeedForwardState,
    LIFState,
    LIFRefracState,
    LIFRefracFeedForwardState,
    lif_refrac_adjoint_step,
    lif_refrac_feed_forward_adjoint_step,
)
from norse.torch.functional.adjoint.lif_refrac_adjoint import (
    lif_refrac_step,
    lif_refrac_feed_forward_step,
)


def test_lif_refrac_adjoint_step():
    input_tensor = torch.ones(1, 10)
    s = LIFRefracState(
        LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10)),
        rho=5 * torch.ones(10),
    )
    input_weights = torch.tensor(np.random.randn(10, 10)).float()
    recurrent_weights = torch.tensor(np.random.randn(10, 10)).float()

    for _ in range(100):
        _, s = lif_refrac_adjoint_step(
            input_tensor, s, input_weights, recurrent_weights
        )


def test_lif_refrac_feed_forward_adjoint_step():
    input_tensor = torch.ones(1, 10)
    s = LIFRefracFeedForwardState(
        LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10)),
        rho=5 * torch.ones(10),
    )

    for _ in range(100):
        _, s = lif_refrac_feed_forward_adjoint_step(input_tensor, s)


def lif_refrac_adjoint_compatibility_test():
    input_tensor = torch.ones(1, 10)
    s0 = LIFRefracState(
        LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10)),
        rho=5 * torch.ones(10),
    )
    s1 = LIFRefracState(
        LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10)),
        rho=5 * torch.ones(10),
    )

    input_weights = torch.tensor(np.random.randn(10, 10)).float()
    recurrent_weights = torch.tensor(np.random.randn(10, 10)).float()

    for _ in range(100):
        z0, s0 = lif_refrac_adjoint_step(
            input_tensor, s0, input_weights, recurrent_weights
        )
        z1, s1 = lif_refrac_step(input_tensor, s1, input_weights, recurrent_weights)

        np.testing.assert_equal(z0.numpy(), z1.numpy())
        np.testing.assert_equal(s0.lif.v.numpy(), s1.lif.v.numpy())
        np.testing.assert_equal(s0.lif.i.numpy(), s1.lif.i.numpy())
        np.testing.assert_equal(s0.rho.numpy(), s1.rho.numpy())


def test_lif_refrac_feed_forward_adjoint_compatibility():
    input_tensor = torch.ones(1, 10)
    s0 = LIFRefracFeedForwardState(
        LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10)),
        rho=5 * torch.ones(10),
    )
    s1 = LIFRefracFeedForwardState(
        LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10)),
        rho=5 * torch.ones(10),
    )

    for _ in range(100):
        z0, s0 = lif_refrac_feed_forward_adjoint_step(input_tensor, s0)
        z1, s1 = lif_refrac_feed_forward_step(input_tensor, s1)

        np.testing.assert_equal(z0.numpy(), z1.numpy())
        np.testing.assert_equal(s0.lif.v.numpy(), s1.lif.v.numpy())
        np.testing.assert_equal(s0.lif.i.numpy(), s1.lif.i.numpy())
        np.testing.assert_equal(s0.rho.numpy(), s1.rho.numpy())
