import math
import torch


def integration_step(a, b, s, t, dt: float = 0.001, sqrtdt: float = math.sqrt(0.001)):
    s_ = s + a(s, t) * dt + b(s, t) * torch.random.normal(mean=0.0, std=sqrtdt)
    return s_
