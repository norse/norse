import torch
from norse.torch.functional.stdp_sensor import stdp_sensor_step, STDPSensorState


def test_sensor():
    s = STDPSensorState(
        a_post=torch.zeros(10, 5),
        a_pre=torch.zeros(5, 10),
    )

    z_pre = torch.ones(10)
    z_post = torch.ones(5)

    s = stdp_sensor_step(z_pre, z_post, s)

    assert s.a_post.shape == (10, 5)
    assert s.a_pre.shape == (5, 10)
