"""
Test exp_filter module and exp_filter function
"""

import torch
from norse.torch.functional.filter import exp_filter_step


def test_exp_filter_step():
    input_data_old_value = torch.tensor([1])
    input_data_new_value = torch.tensor([0])
    filter_parameter = 0.1
    prediction = torch.tensor([0.1])
    actual = exp_filter_step(
        input_data_old_value, input_data_new_value, filter_parameter
    )
    assert torch.allclose(actual, prediction)


def test_exp_filter_step_2():
    input_data_old_value = torch.tensor([1])
    input_data_new_value = torch.tensor([0])
    filter_parameter = 1
    prediction = torch.tensor([1], dtype=torch.float32)
    actual = exp_filter_step(
        input_data_old_value, input_data_new_value, filter_parameter
    )
    assert torch.allclose(actual, prediction)
