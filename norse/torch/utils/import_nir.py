import nir
import nirtorch
import torch

import norse.torch.module.iaf as iaf
import norse.torch.module.lif_box as lif_box
import norse.torch.module.lif as lif


def _import_norse_module(node: nir.NIRNode) -> torch.nn.Module:
    if isinstance(node, nir.Affine):
        return torch.nn.Linear(node.weight.shape[1], node.weight.shape[0])
    if isinstance(node, nir.Conv2d):
        return torch.nn.Conv2d(
            node.weight.shape[1],
            node.weight.shape[0],
            node.kernel_size,
            node.stride,
            node.padding,
            node.dilation,
            node.groups,
        )
    if isinstance(node, nir.Linear):
        return torch.nn.Linear(
            node.weight.shape[1], torch.zeros_like(node.weight.shape[0])
        )
    if isinstance(node, nir.IF):
        return iaf.IAFCell(
            tau=1 / node.tau,  # Invert time constant
            v_threshold=node.v_threshold,
            v_leak=node.v_leak,
            r=torch.ones(node.v_threshold.shape),
        )
    if isinstance(node, nir.CubaLIF):
        return lif.LIFCell(
            tau_mem=1 / node.tau_mem,  # Invert time constant
            tau_syn=1 / node.tau_syn,  # Invert time constant
            v_threshold=node.v_threshold,
            v_leak=node.v_leak,
            r=torch.ones_like(node.v_leak),
        )
    if isinstance(node, nir.LIF):
        return lif_box.LIFBoxCell(
            tau=1 / node.tau,  # Invert time constant
            v_threshold=node.v_threshold,
            v_leak=node.v_leak,
            r=torch.ones_like(node.v_leak),
        )


def from_nir(node: nir.NIRNode) -> torch.nn.Module:
    return nirtorch.load(node, _import_norse_module)
