import pytest
import torch

from norse.torch.functional.stdp import STDPState, stdp_step_conv2d


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_conv2d_stdp_default_parameters(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pre = torch.ones(1, 1, 2, 2, device=device)
    post = torch.zeros(1, 1, 1, 1, device=device)
    weights = torch.full((1, 1, 2, 2), 0.5, device=device)
    state = STDPState(torch.zeros_like(pre), torch.zeros_like(post))

    weights, state = stdp_step_conv2d(pre, post, weights, state)
    weights, state = stdp_step_conv2d(
        torch.zeros_like(pre), torch.ones_like(post), weights, state
    )

    torch.testing.assert_close(
        weights, torch.full_like(weights, 0.5000196), rtol=0, atol=1e-7
    )
    torch.testing.assert_close(state.t_pre, torch.full_like(pre, 0.0196))
    torch.testing.assert_close(state.t_post, torch.full_like(post, 0.02))
