import copy

import torch
from ...functional.stdp import (
    STDPParameters,
    STDPState,
    lif_conv2d_stdp_step,
    lif_linear_stdp_step,
)

from ...functional.lif import LIFFeedForwardState, LIFParameters

# Testing the linear stepper
def test_linear_stdp_stepper(bounds=False):

    # Time points
    time = 1000
    # Layers
    n_batches = 3
    pre, post = 4, 5
    # Weights
    w0 = torch.empty(post, pre)
    torch.nn.init.uniform_(w0, a=0.15, b=0.85)

    # Presynaptic spiking
    z_pre = (torch.rand((time, n_batches, pre)) > 0.1).float()

    # Post synaptic state + parameters
    state_post0 = LIFFeedForwardState(
        v=torch.zeros(n_batches, post),
        i=torch.zeros(n_batches, post),
    )
    p_post = LIFParameters()

    # STDP state
    state_stdp0 = STDPState(
        t_pre=torch.zeros(n_batches, pre),
        t_post=torch.zeros(n_batches, post),
    )

    # Initial weights
    print("Initial weights:")
    print(w0)
    print()

    # Additive
    p_stdp = STDPParameters(
        eta_minus=1e-3,
        eta_plus=1e-3,
        stdp_algorithm="additive",
        hardbound=bounds,
        convolutional=False,
    )
    state_post = copy.deepcopy(state_post0)
    state_stdp = copy.deepcopy(state_stdp0)
    w = copy.deepcopy(w0)
    for i in range(time):
        _, state_post, w, state_stdp = lif_linear_stdp_step(
            z_pre[i], w, state_post, p_post, state_stdp, p_stdp
        )
    print("Additive:")
    print(w)
    print()

    # Additive step
    p_stdp = STDPParameters(
        eta_minus=1e-3,
        eta_plus=1e-3,
        stdp_algorithm="additive_step",
        hardbound=bounds,
        convolutional=False,
    )
    state_post = copy.deepcopy(state_post0)
    state_stdp = copy.deepcopy(state_stdp0)
    w = copy.deepcopy(w0)
    for i in range(time):
        _, state_post, w, state_stdp = lif_linear_stdp_step(
            z_pre[i], w, state_post, p_post, state_stdp, p_stdp
        )
    print("Additive step:")
    print(w)
    print()

    # Multiplicative power
    p_stdp = STDPParameters(
        eta_minus=1e-3,
        eta_plus=1e-3,
        stdp_algorithm="multiplicative_pow",
        mu=0.75,
        hardbound=bounds,
        convolutional=False,
    )
    state_post = copy.deepcopy(state_post0)
    state_stdp = copy.deepcopy(state_stdp0)
    w = copy.deepcopy(w0)
    for i in range(time):
        _, state_post, w, state_stdp = lif_linear_stdp_step(
            z_pre[i], w, state_post, p_post, state_stdp, p_stdp
        )
    print("Multiplicative pow:")
    print(w)
    print()

    # Multiplicative relu
    p_stdp = STDPParameters(
        eta_minus=1e-3,
        eta_plus=1e-3,
        stdp_algorithm="multiplicative_relu",
        hardbound=bounds,
        convolutional=False,
    )
    state_post = copy.deepcopy(state_post0)
    state_stdp = copy.deepcopy(state_stdp0)
    w = copy.deepcopy(w0)
    for i in range(time):
        _, state_post, w, state_stdp = lif_linear_stdp_step(
            z_pre[i], w, state_post, p_post, state_stdp, p_stdp
        )
    print("Multiplicative relu:")
    print(w)
    print()


# Testing the conv2d stepper
def test_conv2d_stdp_stepper(bounds=False):

    # Time points
    time = 1000
    # Architecture (batches + layers + kernel)
    n_batches = 10
    c_pre, c_post = 3, 2
    hw_pre, hw_post = (10, 10), (8, 8)
    kern = (3, 3)

    # Weights
    w0 = torch.nn.Conv2d(c_pre, c_post, kern).weight.detach()
    torch.nn.init.uniform_(w0, a=0.15, b=0.85)

    # Presynaptic spiking
    z_pre = (torch.rand((time, n_batches, c_pre, *hw_pre)) > 0.1).float()

    # Post synaptic state + parameters
    state_post0 = LIFFeedForwardState(
        v=torch.zeros(n_batches, c_post, *hw_post),
        i=torch.zeros(n_batches, c_post, *hw_post),
    )
    p_post = LIFParameters()

    # STDP state
    state_stdp0 = STDPState(
        t_pre=torch.zeros(n_batches, c_pre, *hw_pre),
        t_post=torch.zeros(n_batches, c_post, *hw_post),
    )

    # Initial weights
    print("Initial weights:")
    print(w0[0, 0, :])
    print()

    # Additive
    p_stdp = STDPParameters(
        eta_minus=1e-3,
        eta_plus=1e-3,
        stdp_algorithm="additive",
        hardbound=bounds,
        convolutional=True,
    )
    state_post = copy.deepcopy(state_post0)
    state_stdp = copy.deepcopy(state_stdp0)
    w = copy.deepcopy(w0)
    for i in range(time):
        _, state_post, w, state_stdp = lif_conv2d_stdp_step(
            z_pre[i], w, state_post, p_post, state_stdp, p_stdp
        )
    print("Additive:")
    print(w[0, 0, :])
    print()

    # Additive step
    p_stdp = STDPParameters(
        eta_minus=1e-3,
        eta_plus=1e-3,
        stdp_algorithm="additive_step",
        hardbound=bounds,
        convolutional=True,
    )
    state_post = copy.deepcopy(state_post0)
    state_stdp = copy.deepcopy(state_stdp0)
    w = copy.deepcopy(w0)
    for i in range(time):
        _, state_post, w, state_stdp = lif_conv2d_stdp_step(
            z_pre[i], w, state_post, p_post, state_stdp, p_stdp
        )
    print("Additive step:")
    print(w[0, 0, :])
    print()

    # Multiplicative power
    p_stdp = STDPParameters(
        eta_minus=1e-3,
        eta_plus=1e-3,
        stdp_algorithm="multiplicative_pow",
        mu=0.75,
        hardbound=bounds,
        convolutional=True,
    )
    state_post = copy.deepcopy(state_post0)
    state_stdp = copy.deepcopy(state_stdp0)
    w = copy.deepcopy(w0)
    for i in range(time):
        _, state_post, w, state_stdp = lif_conv2d_stdp_step(
            z_pre[i], w, state_post, p_post, state_stdp, p_stdp
        )
    print("Multiplicative pow:")
    print(w[0, 0, :])
    print()

    # Multiplicative relu
    p_stdp = STDPParameters(
        eta_minus=1e-3,
        eta_plus=1e-3,
        stdp_algorithm="multiplicative_relu",
        hardbound=bounds,
        convolutional=True,
    )
    state_post = copy.deepcopy(state_post0)
    state_stdp = copy.deepcopy(state_stdp0)
    w = copy.deepcopy(w0)
    for i in range(time):
        _, state_post, w, state_stdp = lif_conv2d_stdp_step(
            z_pre[i], w, state_post, p_post, state_stdp, p_stdp
        )
    print("Multiplicative relu:")
    print(w[0, 0, :])
    print()
