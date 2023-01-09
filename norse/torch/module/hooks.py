"""
Hooks to use in torch.nn.Modules (such as `norse.torch.SequentialState`).
"""
from collections import defaultdict

import torch
from torch import prod, nn
from norse.torch.module.snn import SNNBaseClass

_SYNOPS_BUFFER = "n_synops"

# Layers compatible with synops operations
_synops_layers = (
    torch.nn.Linear,
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    SNNBaseClass,  # All SNN modules
)


def _conv_1d_synops(shape, kernel, padding, dilation, stride):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
    return (shape + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1


def _conv_synops(shape: torch.Tensor, module: torch.nn.Module):
    n_dims = len(module.kernel_size)
    synops = 0
    for k in range(n_dims):
        synops += _conv_1d_synops(
            shape[-n_dims + k + 1],
            module.kernel_size[k],
            module.padding[k],
            module.dilation[k],
            module.stride[k],
        )
    return synops


def _conv_transpose_1d_synops(shape, kernel, padding_in, padding_out, dilation, stride):
    return (
        (shape - 1) * stride
        - 2 * padding_in
        + dilation * (kernel - 1)
        + padding_out
        + 1
    )


def _conv_transpose_synops(shape: torch.Tensor, module: torch.nn.Module):
    n_dims = len(module.kernel_size)
    synops = 0
    for k in range(n_dims):
        synops += _conv_transpose_1d_synops(
            shape[-n_dims + k + 1],
            module.kernel_size[k],
            module.padding[k],
            module.output_padding[k],
            module.dilation[k],
            module.stride[k],
        )
    return synops


def synops_forward_hook(module, input, output):
    synops = None
    if (
        isinstance(module, nn.Conv1d)
        or isinstance(module, nn.Conv2d)
        or isinstance(module, nn.Conv3d)
    ):
        synops = _conv_synops(input.shape, module)
    elif (
        isinstance(module, nn.ConvTranspose1d)
        or isinstance(module, nn.ConvTranspose2d)
        or isinstance(module, nn.ConvTranspose3d)
    ):
        synops = _conv_transpose_synops(input.shape, module)
    elif isinstance(module, torch.nn.Linear):
        synops = module.out_features
    elif isinstance(module, SNNBaseClass):
        with torch.no_grad():
            synops = (
                (input if not isinstance(input, tuple) else input[0]).sum().detach()
            )

    if synops is not None:
        module.register_buffer(_SYNOPS_BUFFER, torch.as_tensor(synops))


def count_synops_recursively(module: torch.nn.Module) -> int:
    try:
        synops = module.get_buffer(_SYNOPS_BUFFER)
    except AttributeError:
        synops = 0.0

    for child in module.children():
        synops += count_synops_recursively(child)

    return synops


def register_synops(module):
    if not isinstance(module, torch.nn.Module):
        raise ValueError("Cannot calculate synops on non-module", module)
    elif isinstance(module, _synops_layers):
        try:
            synops = module.get_buffer(_SYNOPS_BUFFER)
        except AttributeError:
            synops = torch.as_tensor(0.0)
            module.register_buffer(_SYNOPS_BUFFER, synops)

        # Register synops in compatible modules
        module.register_forward_hook(synops_forward_hook)

    # Continue recursively
    for child in module.children():
        register_synops(child)
