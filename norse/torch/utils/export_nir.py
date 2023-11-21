from functools import partial
from typing import Optional

import torch
import nir
from nirtorch import extract_nir_graph

import norse.torch.module.iaf as iaf
import norse.torch.module.leaky_integrator_box as leaky_integrator_box
import norse.torch.module.lif as lif
import norse.torch.module.lif_box as lif_box


def _extract_norse_module(
    module: torch.nn.Module, dt: float = 0.001
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
        return nir.CubaLIF(
            tau_mem=dt / module.p.tau_mem_inv.detach(),  # Invert time constant
            tau_syn=dt / module.p.tau_syn_inv.detach(),  # Invert time constant
            v_threshold=module.p.v_th.detach(),
            v_leak=module.p.v_leak.detach(),
            r=torch.ones_like(module.p.v_leak.detach()),
        )
    if isinstance(module, lif_box.LIFBoxCell):
        return nir.LIF(
            tau=dt / module.p.tau_mem_inv.detach(),  # Invert time constant
            v_threshold=module.p.v_th.detach(),
            v_leak=module.p.v_leak.detach(),
            r=torch.ones_like(module.p.v_leak.detach()),
        )
    if isinstance(module, leaky_integrator_box.LIBoxCell):
        return nir.LI(
            tau=dt / module.p.tau_mem_inv.detach(),  # Invert time constant
            v_leak=module.p.v_leak.detach(),
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
    dt: float = 0.001,
) -> nir.NIRNode:
    return extract_nir_graph(
        module,
        partial(_extract_norse_module, dt=dt),
        sample_data,
        model_name=model_name,
    )
