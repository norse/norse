from typing import Optional

import torch
import nir
from nirtorch import extract_nir_graph

from norse.torch.module.leaky_integrator_box import LIBoxCell
from norse.torch.module.lif_box import LIFBoxCell


def _extract_norse_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    if isinstance(module, LIFBoxCell):
        return nir.LIF(
            tau=module.p.tau_mem_inv.detach(),
            v_th=module.p.v_th.detach(),
            v_leak=module.p.v_leak.detach(),
            r=torch.ones_like(module.p.v_leak.detach()),
        )
    if isinstance(module, LIBoxCell):
        return nir.LI(
            tau=module.p.tau_mem_inv.detach(),
            v_leak=module.p.v_leak.detach(),
            r=torch.ones_like(module.p.v_leak.detach()),
        )
    elif isinstance(module, torch.nn.Linear):
        if module.bias is None: # Add zero bias if none is present
            return nir.Linear(
                module.weight.detach(), torch.zeros(*module.weight.shape[:-1])
            )
        else:
            return nir.Linear(module.weight.detach(), module.bias.detach())

    return None


def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "norse"
) -> nir.NIR:
    return extract_nir_graph(
        module, _extract_norse_module, sample_data, model_name=model_name
    )
