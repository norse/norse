from functools import partial
from numbers import Number
from typing import Any, Union

import nir
import nirtorch
import numpy as np
import torch

import norse.torch.functional.reset as reset
import norse.torch.module.iaf as iaf
import norse.torch.module.leaky_integrator_box as li_box
import norse.torch.module.lif_box as lif_box

import logging


def _log_warning(parameter_name: str, expected_value: Any, actual_value: Any):
    logging.warning(
        f"""The parameter {parameter_name} is expected to be set to {expected_value}, but was found to diverge (with mean {actual_value}).
Please read the Norse documentation for more information.
This warning can be turned off by setting from_nir(..., ignore_warnings=True)
"""
    )


def _is_identical(array: np.ndarray, expected_value: float):
    return np.allclose(array, expected_value)


def _to_tensor(tensor: Union[np.ndarray, torch.Tensor]):
    if isinstance(tensor, torch.Tensor):
        return tensor
    if isinstance(tensor, Number):
        return torch.as_tensor(tensor, dtype=torch.float32)
    return torch.from_numpy(tensor).float()


class CubaLIF(torch.nn.Module):
    def __init__(self, w_in, synapse, r, lif):
        super().__init__()
        self.w_in = w_in
        self.synapse = synapse
        self.r = r
        self.lif = lif

    def forward(self, x, state=None):
        if state is None:
            state = (None, None)
        x = self.w_in * x
        x, syn_state = self.synapse(x, state[0])
        x = self.r * x
        z, lif_state = self.lif(x, state[1])
        return z, (syn_state, lif_state)


def _import_norse_module(
    node: nir.NIRNode,
    ignore_warnings: bool = False,
    reset_method: reset.ResetMethod = reset.reset_value,
    dt: float = 0.001,
) -> torch.nn.Module:
    if isinstance(node, nir.Affine):
        has_bias = node.bias is not None
        module = torch.nn.Linear(
            node.weight.shape[1], node.weight.shape[0], bias=has_bias
        )
        module.weight.data = _to_tensor(node.weight)
        if has_bias:
            module.bias.data = _to_tensor(node.bias)
        return module
    if isinstance(node, nir.Conv2d):
        module = torch.nn.Conv2d(
            in_channels=node.weight.shape[1],
            out_channels=node.weight.shape[0],
            kernel_size=node.weight.shape[-2:],
            stride=node.stride,
            padding=node.padding,
            dilation=node.dilation,
            groups=node.groups,
        )
        module.weight.data = _to_tensor(node.weight)
        module.bias.data = _to_tensor(node.bias)
        return module
    if isinstance(node, nir.Flatten):
        return torch.nn.Flatten(node.start_dim, node.end_dim)
    if isinstance(node, nir.Linear):
        module = torch.nn.Linear(
            node.weight.shape[-2], node.weight.shape[-1], bias=False
        )
        module.weight.data = _to_tensor(node.weight)
        return module
    if isinstance(node, nir.IF):
        if not _is_identical(node.r, 1) and not ignore_warnings:
            _log_warning("r", 1, node.r.mean())
        return iaf.IAFCell(iaf.IAFParameters(v_th=_to_tensor(node.v_threshold)), dt=dt)
    if isinstance(node, nir.CubaLIF):
        w_in = _to_tensor(node.w_in)
        synapse = li_box.LIBoxCell(
            li_box.LIBoxParameters(
                tau_mem_inv=1 / _to_tensor(node.tau_syn),  # Invert time constant
                v_leak=_to_tensor(node.v_leak),
            ),
            dt=dt,
        )
        w_rec = _to_tensor(node.r)
        # pytype: disable=wrong-keyword-args
        neuron = lif_box.LIFBoxCell(
            lif_box.LIFBoxParameters(
                tau_mem_inv=1 / _to_tensor(node.tau_mem),  # Invert time constant
                v_th=_to_tensor(node.v_threshold),
                v_leak=_to_tensor(node.v_leak),
                reset_method=reset_method,
            ),
            dt=dt,
        )
        # pytype: enable=wrong-keyword-args
        return CubaLIF(w_in, synapse, w_rec, neuron)

    if isinstance(node, nir.LIF):
        if not _is_identical(node.r, 1) and not ignore_warnings:
            _log_warning("r", 1, _to_tensor(node.r).mean())
        return lif_box.LIFBoxCell(
            lif_box.LIFBoxParameters(
                tau_mem_inv=1 / _to_tensor(node.tau),  # Invert time constant
                v_th=_to_tensor(node.v_threshold),
                v_leak=_to_tensor(node.v_leak),
            ),
            dt=dt,
        )
    if isinstance(node, nir.SumPool2d):
        if not np.allclose(node.padding, 0.0) and not ignore_warnings:
            _log_warning("padding", 0, node.padding.mean())
        return torch.nn.LPPool2d(
            norm_type=1,
            kernel_size=tuple(node.kernel_size),
            stride=tuple(node.stride),
        )
    # if isinstance(node, nir.NIRGraph):
    #     # Currently, just parse a recurrent recurrent Cuba LIF graph
    #     types = {type(v): v for v in node.nodes.values()}
    #     if len(node.nodes) == 4 and nir.CubaLIF in types and nir.Affine in types:
    #         layer_lif = _import_norse_module(types[nir.CubaLIF], ignore_warnings)
    #         layer_affine = _import_norse_module(types[nir.Affine], ignore_warnings)
    #         return sequential.RecurrentSequential(
    #             layer_lif, layer_affine, output_modules=0
    #         )


def from_nir(
    node: nir.NIRNode,
    ignore_warnings: bool = False,
    reset_method: reset.ResetMethod = reset.reset_value,
    dt: float = 0.001,
) -> torch.nn.Module:
    """Converts a NIR graph to a Norse module.

    Example:
    >>> import nir
    >>> import norse.torch
    >>> g = nir.read("my_graph.nir")
    >>> module = norse.torch.from_nir(g)

    Arguments:
        node (nir.NIRNode): The root node of the NIR graph to convert.
        ignore_warnings (bool): Whether to ignore warnings about diverging or unsupported parameters.
        dt (float): The time step of the simulation.
    """
    return nirtorch.load(
        node,
        partial(
            _import_norse_module,
            ignore_warnings=ignore_warnings,
            reset_method=reset_method,
            dt=dt,
        ),
    )
