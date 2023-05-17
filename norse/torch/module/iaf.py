import torch

from norse.torch.module.snn import SNN, SNNCell
from norse.torch.functional.iaf import (
    iaf_feed_forward_step,
    IAFFeedForwardState,
    IAFParameters,
)


class IAFCell(SNNCell):
    def __init__(self, p: IAFParameters = IAFParameters(), dt: float = 0.001):
        r"""Feedforward step of an integrate-and-fire neuron, computing a single step

        .. math::
            \dot{v} = v

        together with the jump condition

        .. math::
            z = \Theta(v - v_{\text{th}})

        and transition equation

        .. math::
            v = (1-z) v + z v_{\text{reset}}

        Parameters:
            p (IAFParameters): parameters of a leaky integrate and fire neuron
            dt (float): Integration timestep to use (unused, but added for compatibility)
        """
        super().__init__(iaf_feed_forward_step, self.initial_state, p, dt)

    def initial_state(self, input_tensor: torch.Tensor) -> IAFFeedForwardState:
        state = IAFFeedForwardState(
            # v=torch.full(
            #     input_tensor.shape,
            #     self.p.v_reset.detach(),
            #     device=input_tensor.device,
            #     dtype=torch.float32,
            # ),
            v=self.p.v_reset.to(input_tensor.device)
        )
        state.v.requires_grad = True
        return state


class IAF(SNN):
    def __init__(self, p: IAFParameters = IAFParameters(), dt: float = 0.001):
        r"""Feedforward step of an integrate-and-fire neuron, computing a single step

        .. math::
            \dot{v} = v

        together with the jump condition

        .. math::
            z = \Theta(v - v_{\text{th}})

        and transition equation

        .. math::
            v = (1-z) v + z v_{\text{reset}}

        Parameters:
            p (IAFParameters): parameters of a leaky integrate and fire neuron
            dt (float): Integration timestep to use (unused, but added for compatibility)
        """
        super().__init__(iaf_feed_forward_step, self.initial_state, p, dt)

    def initial_state(self, input_tensor: torch.Tensor) -> IAFFeedForwardState:
        state = IAFFeedForwardState(
            # v=torch.full(
            #     input_tensor.shape,
            #     self.p.v_reset.detach(),
            #     device=input_tensor.device,
            #     dtype=torch.float32,
            # ),
            v=self.p.v_reset
        )
        state.v.requires_grad = True
        return state
