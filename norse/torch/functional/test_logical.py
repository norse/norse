import torch
import numpy as np

from .logical import logical_and, logical_or, logical_xor, muller_c, posedge_detector


def logical_and_test():
    z = logical_and(torch.tensor([1, 0, 0, 1]), torch.tensor([1, 0, 0, 0]))
    np.testing.assert_equal(z.numpy(), np.array([1, 0, 0, 0]))


def logical_or_test():
    z = logical_or(torch.tensor([1, 0, 0, 1]), torch.tensor([1, 0, 0, 0]))
    np.testing.assert_equal(z.numpy(), np.array([1, 0, 0, 1]))


def logical_xor_test():
    z = logical_xor(torch.tensor([1, 0, 0, 1]), torch.tensor([1, 0, 0, 0]))
    np.testing.assert_equal(z.numpy(), np.array([0, 0, 0, 1]))


def posedge_detector_test():
    z = torch.tensor([1, 0, 0, 1])
    z_prev = torch.tensor([0, 0, 1, 1])
    z_out = posedge_detector(z, z_prev)
    np.testing.assert_equal(z_out.numpy(), np.array([1, 0, 0, 0]))
