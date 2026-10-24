import pytest
import torch

from norse.torch.module.lif import LIFCell
from norse.torch.module.regularization import RegularizationCell
from norse.torch.functional.regularization import voltage_accumulator


def test_regularization_module():
    cell = LIFCell()
    r = RegularizationCell()  # Defaults to spike counting
    data = torch.ones(5, 2) + 10  # Batch size of 5
    z, s = cell(data)
    z, rs = r(z, s)
    assert z.shape == (5, 2)
    assert rs == 10
    z, s = cell(data, s)
    z, rs = r(z, s)
    assert rs == 20
    assert r.state == 20


def test_regularization_module_reset_allows_new_batch_size():
    """RegularizationCell.state is cached from the first call via
    torch.zeros_like(s.v) (voltage_accumulator), so a later call with a
    different batch size must be rejected with a clear error, and reset()
    must be the documented way to recover and reuse the cell with the new
    shape, instead of only being able to construct a brand-new instance."""
    cell = LIFCell()
    r = RegularizationCell(accumulator=voltage_accumulator)
    assert hasattr(
        r, "reset"
    ), "RegularizationCell must expose reset(), like SynOpsCounter"

    data5 = torch.ones(5, 2) + 10
    z, s = cell(data5)
    z, rs = r(z, s)
    assert rs.shape == (5, 2)

    data3 = torch.ones(3, 2) + 10
    z3, s3 = cell(data3)
    with pytest.raises(ValueError):
        r(z3, s3)

    r.reset()
    assert r.state is None
    z3, rs3 = r(z3, s3)
    assert rs3.shape == (3, 2)
