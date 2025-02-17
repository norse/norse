from functools import partial
from typing import Optional
import logging

import torch
import nir
import numpy as np
from nirtorch import extract_nir_graph

import norse.torch.module.iaf as iaf
import norse.torch.module.leaky_integrator_box as leaky_integrator_box
import norse.torch.module.lif as lif
import norse.torch.module.lif_box as lif_box


def _align_shapes(
    a: torch.Tensor, shape: torch.Size, message: str = ""
) -> torch.Tensor:
    if not a.shape == shape:
        try:
            return a.view(shape)
        except RuntimeError:
            logging.error(
                f"Could not align shapes {a.shape} and {b.shape} of parameter {message}"
            )
    else:
        return a


def _extract_norse_module(
    module: torch.nn.Module, dt: float = 1
) -> Optional[nir.NIRNode]:
    if isinstance(module, torch.nn.Conv2d):
        return nir.Conv2d(
            input_shape=None,
            weight=module.weight.detach(),
            bias=module.bias.detach(),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
    if isinstance(module, lif.LIFCell):
        shape = module.p.tau_mem_inv.shape
        return nir.CubaLIF(
            tau_mem=dt
            / _align_shapes(
                module.p.tau_mem_inv.detach(), shape, "tau_syn"
            ),  # Invert time constant
            tau_syn=dt
            / _align_shapes(
                module.p.tau_syn_inv.detach(), shape, "tau_syn"
            ),  # Invert time constant
            v_threshold=_align_shapes(module.p.v_th.detach(), shape, "v_th"),
            v_leak=_align_shapes(module.p.v_leak.detach(), shape, "v_leak"),
            r=np.ones_like(module.p.v_leak.detach()),
            w_in=np.ones_like(module.p.v_leak.detach()),
        )
    if isinstance(module, lif_box.LIFBoxCell):
        shape = module.p.tau_mem_inv.shape
        return nir.LIF(
            tau=dt
            / _align_shapes(
                module.p.tau_mem_inv.detach(), shape, "tau"
            ),  # Invert time constant
            v_threshold=_align_shapes(module.p.v_th.detach(), shape, "v_th"),
            v_leak=_align_shapes(module.p.v_leak.detach(), shape, "v_leak"),
            r=torch.ones_like(module.p.v_leak.detach()),
        )
    if isinstance(module, leaky_integrator_box.LIBoxCell):
        shape = module.p.tau_mem_inv.shape
        return nir.LI(
            tau=dt
            / _align_shapes(
                module.p.tau_mem_inv.detach(), shape, "tau_mem_inv"
            ),  # Invert time constant
            v_leak=_align_shapes(module.p.v_leak.detach(), shape, "v_leak"),
            r=torch.ones_like(module.p.v_leak.detach()),
        )
    if isinstance(module, iaf.IAFCell):
        return nir.IF(
            r=torch.ones_like(module.p.v_th.detach()),
            v_threshold=module.p.v_th.detach(),
        )
    elif isinstance(module, torch.nn.Linear):
        if module.bias is None:  # Add zero bias if none is present
            return nir.Affine(
                module.weight.detach(), torch.zeros(*module.weight.shape[:-1])
            )
        else:
            return nir.Affine(module.weight.detach(), module.bias.detach())

    return None


def to_nir(
    module: torch.nn.Module,
    sample_data: torch.Tensor,
    model_name: str = "norse",
    dt: float = 1,
) -> nir.NIRNode:
    """Converts a Norse module to a NIR graph.

    Args:
        module: Norse module to convert
        sample_data: Sample data to infer input shape
        model_name: Name of the model
        dt: Integration time step to use when converting neurons. This parameter should be set to 1
            to retain the dynamics of the neuron equation. However, if your network has been trained
            with a different dt, you should set this parameter to the value used during training.
    """
    return extract_nir_graph(
        module,
        partial(_extract_norse_module, dt=dt),
        sample_data,
        model_name=model_name,
    )
