from typing import Optional

import torch
import nir
from nirtorch import extract_nir_graph

from norse.torch.module.lif_box import LIFBoxCell


def _extract_norse_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    if isinstance(module, LIFBoxCell):
        return nir.LIF(
            tau=module.p.tau_mem_inv,
            v_th=module.p.v_th,
            v_leak=module.p.v_leak,
            r=torch.ones_like(module.p.v_leak),
        )
    elif isinstance(module, torch.nn.Linear):
        return nir.Linear(module.weight, module.bias)

    return None


def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "norse"
) -> nir.NIR:
    return extract_nir_graph(
        module, _extract_norse_module, sample_data, model_name=model_name
    )
