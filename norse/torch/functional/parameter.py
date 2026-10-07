import torch
import copy


def _float_parameter(value):
    """Promote integer parameters before scalar arithmetic for ONNX export.

    Keep floating tensors unchanged so time-step scaling can happen before
    multiplication with the state, avoiding overflow in low-precision states.
    """
    if isinstance(value, torch.Tensor):
        if not (value.is_floating_point() or value.is_complex()):
            return value.to(dtype=torch.get_default_dtype())
    return value


DEFAULT_BIO_PARAMS = {
    "iaf": {"v_reset": torch.tensor(-70.0), "v_th": torch.as_tensor(-55.0)},
    "li": {
        "tau_syn_inv": torch.as_tensor(1 / 2.0),
        "tau_mem_inv": torch.as_tensor(1 / 10.0),
        "v_leak": torch.as_tensor(-70.0),
    },
    "lif": {
        "tau_syn_inv": torch.as_tensor(1 / 2.0),
        "tau_mem_inv": torch.as_tensor(1 / 10.0),
        "v_leak": torch.as_tensor(-70.0),
        "v_th": torch.as_tensor(-55.0),
        "v_reset": torch.as_tensor(-70.0),
    },
    "lifAdEx": {
        "adaptation_current": torch.tensor(0),
        "adaptation_spike": torch.tensor(0.01),
        "delta_T": torch.tensor(2.0),
        "tau_ada_inv": torch.as_tensor(1 / 100.0),
        "tau_syn_inv": torch.as_tensor(1 / 2.0),
        "tau_mem_inv": torch.as_tensor(1 / 10.0),
        "v_leak": torch.as_tensor(-70.0),
        "v_th": torch.as_tensor(-55.0),
        "v_reset": torch.as_tensor(-70.0),
    },
    "lifEx": {
        "delta_T": torch.tensor(0),
        "tau_syn_inv": torch.as_tensor(1 / 2.0),
        "tau_mem_inv": torch.as_tensor(1 / 10.0),
        "v_leak": torch.as_tensor(-70.0),
        "v_th": torch.as_tensor(-55.0),
        "v_reset": torch.as_tensor(-70.0),
    },
    "lsnn": {
        "tau_syn_inv": torch.as_tensor(1 / 2.0),
        "tau_mem_inv": torch.as_tensor(1 / 10.0),
        "tau_adapt_inv": torch.as_tensor(1 / 100.0),
        "v_leak": torch.as_tensor(-70.0),
        "v_th": torch.as_tensor(-55.0),
        "v_reset": torch.as_tensor(-70.0),
        "beta": torch.tensor(0.5),
    },
}


def default_bio_parameters(model, **kwargs):
    params = copy.deepcopy(DEFAULT_BIO_PARAMS[model])
    params.update(kwargs)
    return params
