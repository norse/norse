import torch
from norse.torch.module.exp_filter import ExpFilter


def test_ExpFilter_forward_parameter_one_and_ones_weights_without_bias():
    # parameter = 1; weights = 1; without bias
    tau_filter_inv = 0
    feature_size = 1
    output_size = 1
    init_weights = torch.ones(feature_size, output_size)
    filter_layer = ExpFilter(
        feature_size,
        output_size,
        tau_filter_inv=tau_filter_inv,
        input_weights=init_weights,
        bias=False,
    )
    data = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
    data = data.reshape(data.shape[0], 1, 1)
    prediction = torch.ones(data.shape[0])
    prediction = prediction.reshape(prediction.shape[0], 1, 1)
    actual = filter_layer(data)
    assert torch.allclose(actual, prediction)


def test_ExpFilter_forward_ones_weights_without_bias():
    # parameter = 1e-3; weights = 1; without bias
    feature_size = 1
    output_size = 1
    init_weights = torch.ones(feature_size, output_size)
    tau_filter_inv = 1  # => parameter = exp(-1e-3)
    filter_layer = ExpFilter(
        feature_size,
        output_size,
        tau_filter_inv=tau_filter_inv,
        input_weights=init_weights,
        bias=False,
    )
    model_parameter = torch.exp(torch.tensor([-1e-3])).item()
    data = torch.tensor([1, 0, 0], dtype=torch.float32).reshape(3, 1, 1)
    prediction = torch.tensor([1, model_parameter, model_parameter**2]).reshape(3, 1, 1)
    actual = filter_layer(data)
    for name, param in filter_layer.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    assert torch.allclose(actual, prediction)


def test_ExpFilter_forward_ones_weights():
    # parameter = 1e-3; weights = 1; with bias
    feature_size = 1
    output_size = 1
    init_weights = torch.ones(feature_size, output_size)
    tau_filter_inv = 1  # => parameter = exp(-1e-3)
    filter_layer = ExpFilter(
        feature_size,
        output_size,
        tau_filter_inv=tau_filter_inv,
        input_weights=init_weights,
        bias=True,
    )
    for name, param in filter_layer.named_parameters():
        if name == "linear.bias":
            bias = param.item()
    model_parameter = torch.exp(torch.tensor([-1e-3])).item()
    data = torch.tensor([1, 0, 0], dtype=torch.float32).reshape(3, 1, 1)
    prediction = torch.tensor([1 + bias, 0, 0], dtype=torch.float32).reshape(3, 1, 1)

    for i in range(prediction.shape[0] - 1):
        prediction[i + 1] += prediction[i] * model_parameter + bias
    actual = filter_layer(data)

    assert torch.allclose(actual, prediction)


def test_ExpFilterforward_():
    # parameter = 1e-3; weights = random; with bias
    feature_size = 1
    output_size = 1
    tau_filter_inv = 1  # => parameter = exp(-1e-3)
    filter_layer = ExpFilter(
        feature_size, output_size, tau_filter_inv=tau_filter_inv, bias=True
    )
    for name, param in filter_layer.named_parameters():
        if name == "linear.weight":
            weight = param.item()
        if name == "linear.bias":
            bias = param.item()
    model_parameter = torch.exp(torch.tensor([-1e-3])).item()
    data = torch.tensor([1, 2, 3], dtype=torch.float32).reshape(3, 1, 1)
    data_after_linear_layer = data * weight + bias
    prediction = torch.zeros_like(data)
    prediction[0] = data_after_linear_layer[0]

    for i in range(prediction.shape[0] - 1):
        prediction[i + 1] += (
            prediction[i] * model_parameter + data_after_linear_layer[i + 1]
        )
    actual = filter_layer(data)
    assert torch.allclose(actual, prediction)
