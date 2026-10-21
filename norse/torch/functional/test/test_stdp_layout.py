import pytest
import torch

from norse.torch.functional.stdp import STDPParameters, STDPState, stdp_step_conv2d


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "algorithm",
    ["additive", "additive_step", "multiplicative_pow", "multiplicative_relu"],
)
@pytest.mark.parametrize("contiguous", [False, True])
def test_conv_stdp_is_invariant_to_spatial_tensor_layout(dtype, algorithm, contiguous):
    pre = (torch.arange(2 * 3 * 5 * 4).reshape(2, 3, 5, 4) % 3 == 0).to(dtype)
    post = (torch.arange(2 * 2 * 4 * 3).reshape(2, 2, 4, 3) % 3 == 1).to(dtype)
    pre = pre.transpose(-1, -2)
    post = post.transpose(-1, -2)
    pre_trace = pre * 0.2 + 0.1
    post_trace = post * 0.3 + 0.2
    weights = torch.linspace(0.2, 0.7, 24, dtype=dtype).reshape(2, 3, 2, 2)
    params = STDPParameters(
        eta_plus=0.03,
        eta_minus=0.01,
        stdp_algorithm=algorithm,
        mu=0.75,
        convolutional=True,
        hardbound=False,
    )
    expected_weights, expected_state = stdp_step_conv2d(
        pre.contiguous(),
        post.contiguous(),
        weights,
        STDPState(pre_trace.contiguous(), post_trace.contiguous()),
        params,
    )
    if contiguous:
        pre, post = pre.contiguous(), post.contiguous()
        pre_trace, post_trace = pre_trace.contiguous(), post_trace.contiguous()
    else:
        assert not post.is_contiguous() and not post_trace.is_contiguous()

    actual_weights, actual_state = stdp_step_conv2d(
        pre, post, weights, STDPState(pre_trace, post_trace), params
    )

    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(actual_state.t_pre, expected_state.t_pre)
    torch.testing.assert_close(actual_state.t_post, expected_state.t_post)
