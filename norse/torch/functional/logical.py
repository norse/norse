import torch

from norse.torch.functional.superspike import super_fn


def logical_and(x, y):
    """Computes a logical and provided x and y are bitvectors."""
    return x * y


def logical_xor(x, y):
    """Computes a logical xor provided x and y are bitvectors."""
    return torch.abs(x - y)


def logical_or(x, y):
    """Computes a logical or provided x and y are bitvectors."""
    return super_fn(x + y)


def muller_c(y_prev, x_1, x_2):
    """Computes the muller-c element next state provided x_1 and x_2 are bitvectors
    and y_prev is the previous state."""
    return super_fn(x_1 * x_2 + (x_1 + x_2) * y_prev)


def posedge_detector(z, z_prev):
    """Determines whether a transition from 0 to 1 has occured
    providing that z and z_prev are bitvectors
    """
    return (1 - z_prev) * z
