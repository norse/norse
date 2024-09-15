import torch
import norse


def test_super_threshold():
    m = norse.torch.SpikeThreshold()
    t = torch.tensor([-1, -0.01, 0, 0.01, 1, 2])
    assert torch.all(
        torch.eq(
            m(t),
            norse.torch.functional.threshold.threshold(
                t - 1, method="super", alpha=100.0
            ),
        )
    )
