import typing
import logging
import warnings

import torch
import nir
import numpy as np
import nirtorch

import norse


def _align_shapes(
    a: torch.Tensor, shape: torch.Size, message: str = ""
) -> torch.Tensor:
    if a.numel() == 1:
        return a.repeat(shape)
    if not a.shape == shape:
        try:
            return a.view(shape)
        except RuntimeError:
            logging.error(
                f"Could not align shapes {a.shape} and {shape} of parameter {message}"
            )
    else:
        return a


def _norse_to_nir_mapping_dict(
    time_scaling_factor: float = 1,
) -> typing.Dict[torch.nn.Module, typing.Callable[[torch.nn.Module], nir.NIRNode]]:
    norse_map = {}

    # Projections

    def _map_conv2d(module: torch.nn.Conv2d):
        return nir.Conv2d(
            input_shape=None,
            weight=module.weight.detach().numpy(),
            bias=module.bias.detach().numpy() if module.bias is not None else np.zeros(module.weight.shape[0]),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    norse_map[torch.nn.Conv2d] = _map_conv2d

    def _map_conv1d(module: torch.nn.Conv1d):
        return nir.Conv1d(
            input_shape=None,
            weight=module.weight.detach().numpy(),
            bias=module.bias.detach().numpy() if module.bias is not None else np.zeros(module.weight.shape[0]),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    norse_map[torch.nn.Conv1d] = _map_conv1d

    def _map_linear(module: torch.nn.Linear):
        if module.bias is None:
            return nir.Linear(module.weight.detach().numpy())
        else:
            return nir.Affine(module.weight.detach().numpy(), module.bias.detach().numpy())

    norse_map[torch.nn.Linear] = _map_linear

    def _map_flatten(module: torch.nn.Flatten):
        # The Flatten node usually skips one dimension and starts at dimension 1.
        # This is because the batch dimension is not being flattened. However, NIR
        # doesn't include the batch dimension in its input/output shapes, which is
        # why we subtract 1 from the start dimension and end dimension.
        return nir.Flatten(
            input_type=None,
            start_dim=module.start_dim - 1 if module.start_dim > 0 else module.end_dim,
            end_dim=module.end_dim - 1 if module.end_dim > 0 else module.end_dim
        )

    norse_map[torch.nn.Flatten] = _map_flatten

    def _map_avgpool2d(module: torch.nn.AvgPool2d):
        return nir.AvgPool2d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding
        )

    norse_map[torch.nn.AvgPool2d] = _map_avgpool2d

    # Neurons

    def _map_cuba_lif(module: norse.torch.LIFCell):
        shape = module.p.tau_mem_inv.shape
        return nir.CubaLIF(
            tau_mem=time_scaling_factor
            / _align_shapes(
                module.p.tau_mem_inv.detach(), shape, "tau_mem"
            ).numpy(),  # Invert time constant
            tau_syn=time_scaling_factor
            / _align_shapes(
                module.p.tau_syn_inv.detach(), shape, "tau_syn"
            ).numpy(),  # Invert time constant
            v_reset=_align_shapes(module.p.v_reset.detach(), shape, "v_reset").numpy(),
            v_threshold=_align_shapes(module.p.v_th.detach(), shape, "v_th").numpy(),
            v_leak=_align_shapes(module.p.v_leak.detach(), shape, "v_leak").numpy(),
            r=np.ones_like(module.p.v_leak.detach().numpy()),
            w_in=np.ones_like(module.p.v_leak.detach().numpy()),
        )

    norse_map[norse.torch.LIFCell] = _map_cuba_lif

    def _map_leaky_integrator(module: norse.torch.LICell):
        shape = module.p.tau_mem_inv.shape
        return nir.CubaLI(
            tau_mem=time_scaling_factor
            / _align_shapes(
                module.p.tau_mem_inv.detach(), shape, "tau_syn"
            ).numpy(),  # Invert time constant
            tau_syn=time_scaling_factor
            / _align_shapes(
                module.p.tau_syn_inv.detach(), shape, "tau_syn"
            ).numpy(),  # Invert time constant
            v_leak=_align_shapes(module.p.v_leak.detach(), shape, "v_leak").numpy(),
            r=np.ones_like(module.p.v_leak.detach().numpy()),
            w_in=np.ones_like(module.p.v_leak.detach().numpy()),
        )

    norse_map[norse.torch.LICell] = _map_leaky_integrator

    def _map_lif_box(module: norse.torch.LIFBoxCell):
        shape = module.p.tau_mem_inv.shape
        return nir.LIF(
            tau=time_scaling_factor
            / _align_shapes(
                module.p.tau_mem_inv.detach(), shape, "tau"
            ).numpy(),  # Invert time constant
            v_threshold=_align_shapes(module.p.v_th.detach(), shape, "v_th").numpy(),
            v_leak=_align_shapes(module.p.v_leak.detach(), shape, "v_leak").numpy(),
            r=np.ones_like(module.p.v_leak.detach().numpy()),
            v_reset=_align_shapes(module.p.v_reset.detach(), shape, "v_reset").numpy(),
        )

    norse_map[norse.torch.LIFBoxCell] = _map_lif_box

    def _map_li_box(module: norse.torch.LIBoxCell):
        shape = module.p.tau_mem_inv.shape
        return nir.LI(
            tau=time_scaling_factor
            / _align_shapes(
                module.p.tau_mem_inv.detach(), shape, "tau_mem_inv"
            ).numpy(),  # Invert time constant
            v_leak=_align_shapes(module.p.v_leak.detach(), shape, "v_leak").numpy(),
            r=np.ones_like(module.p.v_leak.detach().numpy()),
        )

    norse_map[norse.torch.LIBoxCell] = _map_li_box

    def _map_iaf(module: norse.torch.IAFCell):
        return nir.IF(
            r=np.ones_like(module.p.v_th.detach().numpy()),
            v_threshold=module.p.v_th.detach().numpy(),
            v_reset=_align_shapes(
                module.p.v_reset.detach(), module.p.v_th.shape, "v_reset"
            ).numpy(),
        )

    norse_map[norse.torch.IAFCell] = _map_iaf

    return norse_map


def to_nir(
    module: torch.nn.Module,
    sample_data: typing.Optional[torch.Tensor] = None,
    model_name: str = "norse",
    time_scaling_factor: float = 1,
    type_check: bool = True,
) -> nir.NIRNode:
    """Converts a Norse module to a NIR graph.

    Args:
        module (torch.nn.Module): Norse module to convert
        sample_data (torch.Tensor): Sample data to infer input shape. DEPRECATED and will be removed in
            future releases.
        model_name (str): Name of the model. DEPRECATED and will be removed in future releases
        time_scaling_factor (float): Integration time step to use when converting neurons. This parameter
            defaults to 1 which retains the dynamics of the neuron equation. However, if your network has
            been trained with a different dt, this scaling factor can re-scale the network dynamics as
            needed.
        type_check (bool): Whether to run type checking on generated NIRGraphs
    """
    if sample_data is not None:
        warnings.warn(
            "The 'sample_data' parameter is deprecated and unused, it will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )

    if model_name != "norse":
        warnings.warn(
            "The 'model_name' parameter is deprecated and unused, it will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )

    mapping_dict = _norse_to_nir_mapping_dict(time_scaling_factor=time_scaling_factor)

    # Define stateful modules that return (output, state) tuples
    # These modules need special handling during tracing
    stateful_modules = {
        norse.torch.LIFCell,
        norse.torch.LIFBoxCell,
        norse.torch.LIBoxCell,
        norse.torch.IAFCell,
    }

    return nirtorch.torch_to_nir(
        module=module,
        module_map=mapping_dict,
        type_check=type_check,
        stateful_modules=stateful_modules,
        concrete_args={"state": None},
    )
