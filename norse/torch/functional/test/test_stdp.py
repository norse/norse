import pytest
import torch

from ...functional.stdp import (
    STDPParameters,
    STDPState,
    stdp_step_linear,
    stdp_step_conv2d,
)


def create_id(param):
    return param[0]


# Linear STDP
@pytest.fixture(
    scope="function",
    params=[
        ["additive", 0.0],
        ["additive_step", 0.0],
        ["multiplicative_pow", 0.75],
        ["multiplicative_relu", 1.0],
    ],
    ids=create_id,
)
def initialise_for_linear_stdp(request):
    stdp_algorithm = request.param[0]
    n_batches = 1
    n_pre, n_post = 2, 3
    w0 = 0.5 * torch.ones(n_post, n_pre)
    z_pre = torch.tensor(
        [
            [[1.0, 1.0]],
            [[0.0, 0.0]],
            [[1.0, 1.0]],
        ]
    ).float()
    z_post = torch.tensor(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0]],
        ]
    ).float()
    state_stdp = STDPState(
        t_pre=torch.zeros(n_batches, n_pre),
        t_post=torch.zeros(n_batches, n_post),
    )
    p_stdp = STDPParameters(
        eta_minus=1e-1,
        eta_plus=3e-1,  # Best to check with large, asymmetric learning-rates
        stdp_algorithm=stdp_algorithm,
        mu=request.param[1],
        hardbound=True,
        convolutional=False,
    )

    return (
        stdp_algorithm,
        n_batches,
        n_pre,
        n_post,
        w0,
        z_pre,
        z_post,
        state_stdp,
        p_stdp,
    )


def test_linear_stdp_stepper(initialise_for_linear_stdp):
    (
        _,
        n_batches,
        n_pre,
        n_post,
        w,
        z_pre,
        z_post,
        state_stdp,
        p_stdp,
    ) = initialise_for_linear_stdp

    n_time = z_pre.shape[0]

    t_pre = 0.0
    t_post = 0.0
    for n_t in range(n_time):
        w0 = w
        w, state_stdp = stdp_step_linear(
            z_pre[n_t],
            z_post[n_t],
            w,
            state_stdp,
            p_stdp,
            dt=0.001,
        )

        # Calculating the gradient for one synapse
        t_pre += (
            0.001 * (p_stdp.tau_pre_inv) * (-t_pre + p_stdp.a_pre * z_pre[n_t][0][0])
        )
        t_post += (
            0.001
            * (p_stdp.tau_post_inv)
            * (-t_post + p_stdp.a_post * z_post[n_t][0][0])
        )

        # Check potentiation
        if n_t == 1:
            assert torch.allclose(
                torch.abs(w - w0) / torch.pow(p_stdp.w_max - w0, p_stdp.mu),
                p_stdp.eta_plus * t_pre,
                atol=1e-6,
                rtol=0.0,
            )
        # Check depression
        if n_t == 2:
            assert torch.allclose(
                torch.abs(w - w0) / torch.pow(w0, p_stdp.mu),
                p_stdp.eta_minus * t_post,
                atol=1e-6,
                rtol=0.0,
            )

    # Shape checks
    assert state_stdp.t_pre.shape == (n_batches, n_pre)
    assert state_stdp.t_post.shape == (n_batches, n_post)
    assert w.shape == (n_post, n_pre)


# Conv2D STDP
@pytest.fixture(
    scope="function",
    params=[
        ["additive", 0.0],
        ["additive_step", 0.0],
        ["multiplicative_pow", 0.75],
        ["multiplicative_relu", 1.0],
    ],
    ids=create_id,
)
def initialise_for_conv2d_stdp(request):
    stdp_algorithm = request.param[0]
    n_batches = 1
    c_pre, c_post = 3, 2
    hw_pre, hw_post = (10, 10), (8, 8)
    hw_kern = (3, 3)
    w0 = torch.nn.Conv2d(c_pre, c_post, *hw_kern).weight.detach()
    torch.nn.init.constant_(w0, 0.5)
    z_pre = torch.stack(
        (
            torch.ones(n_batches, c_pre, *hw_pre),
            torch.zeros(n_batches, c_pre, *hw_pre),
            torch.ones(n_batches, c_pre, *hw_pre),
        ),
        dim=0,
    )
    z_post = torch.stack(
        (
            torch.zeros(n_batches, c_post, *hw_post),
            torch.ones(n_batches, c_post, *hw_post),
            torch.zeros(n_batches, c_post, *hw_post),
        ),
        dim=0,
    )
    state_stdp = STDPState(
        t_pre=torch.zeros(n_batches, c_pre, *hw_pre),
        t_post=torch.zeros(n_batches, c_post, *hw_post),
    )
    p_stdp = STDPParameters(
        eta_minus=1e-2,
        eta_plus=3e-2,  # Best to check with large, asymmetric learning-rates
        stdp_algorithm=stdp_algorithm,
        mu=request.param[1],
        hardbound=False,
        convolutional=True,
    )

    return (
        stdp_algorithm,
        n_batches,
        c_pre,
        c_post,
        hw_pre,
        hw_post,
        hw_kern,
        w0,
        z_pre,
        z_post,
        state_stdp,
        p_stdp,
    )


def test_conv2d_stdp_stepper(initialise_for_conv2d_stdp):
    (
        _,
        n_batches,
        c_pre,
        c_post,
        hw_pre,
        hw_post,
        hw_kern,
        w,
        z_pre,
        z_post,
        state_stdp,
        p_stdp,
    ) = initialise_for_conv2d_stdp

    n_time = z_pre.shape[0]

    t_pre = 0.0
    t_post = 0.0
    for n_t in range(n_time):
        w0 = w
        w, state_stdp = stdp_step_conv2d(
            z_pre[n_t],
            z_post[n_t],
            w,
            state_stdp,
            p_stdp,
            dt=0.001,
        )

        # Calculating the gradient for one synapse
        t_pre += (
            0.001
            * p_stdp.tau_pre_inv
            * (-t_pre + p_stdp.a_pre * z_pre[n_t][0][0][0][0])
        )
        t_post += (
            0.001
            * p_stdp.tau_post_inv
            * (-t_post + p_stdp.a_post * z_post[n_t][0][0][0][0])
        )

        # Check potentiation
        if n_t == 1:
            assert torch.allclose(
                torch.abs(w - w0) / torch.pow(p_stdp.w_max - w0, p_stdp.mu),
                hw_post[0] * hw_post[1] * p_stdp.eta_plus * t_pre,
                atol=1e-6,
                rtol=0.0,
            )

        # Check depression
        if n_t == 2:
            assert torch.allclose(
                torch.abs(w - w0) / torch.pow(w0 - p_stdp.w_min, p_stdp.mu),
                hw_post[0] * hw_post[1] * p_stdp.eta_minus * t_post,
                atol=1e-6,
                rtol=0.0,
            )

    # Shape checks
    assert state_stdp.t_pre.shape == (n_batches, c_pre, *hw_pre)
    assert state_stdp.t_post.shape == (n_batches, c_post, *hw_post)
    assert w.shape == (c_post, c_pre, *hw_kern)
