import torch

from norse.torch.functional.lif_adex import (
    LIFAdExState,
    LIFAdExFeedForwardState,
    LIFAdExParameters,
)

from norse.torch.functional.lif_adex_refrac import (
    LIFAdExRefracParameters,
    LIFAdExRefracState,
    LIFAdExRefracFeedForwardState,
    lif_adex_refrac_step,
    lif_adex_refrac_feed_forward_step,
)

from norse.torch.module.snn import SNN, SNNCell, SNNRecurrentCell, SNNRecurrent


class LIFAdExRefracCell(SNNCell):
    def __init__(
        self, p: LIFAdExRefracParameters = LIFAdExRefracParameters(), **kwargs
    ) -> None:
        super().__init__(
            lif_adex_refrac_feed_forward_step,
            self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(
        self, input_tensor: torch.Tensor
    ) -> LIFAdExRefracFeedForwardState:
        state = LIFAdExRefracFeedForwardState(
            LIFAdExFeedForwardState(
                v=self.p.v_leak.detach(),
                i=torch.zeros(
                    *input_tensor.shape,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                a=torch.zeros(
                    *input_tensor.shape,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            ),
            rho=torch.zeros(
                input_tensor.shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.lif_adex.v.requires_grad = True
        return state


class LIFAdExRefracRecurrentCell(SNNRecurrentCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFAdExRefracParameters,
        **kwargs,
    ) -> None:
        super().__init__(
            activation=lif_adex_refrac_step,
            state_fallback=self.initial_state,
            p=p,
            input_size=input_size,
            hidden_size=hidden_size,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFAdExRefracState:
        dims = (*input_tensor.shape[:-1], self.hidden_size)
        state = LIFAdExRefracState(
            LIFAdExState(
                z=torch.zeros(
                    *dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                v=torch.full(
                    dims,
                    self.p.v_leak.detach(),
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                i=torch.zeros(
                    *dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                a=torch.zeros(
                    *dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            ),
            rho=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.lif_adex.v.requires_grad = True
        return state


class LIFAdExRefracRecurrent(SNNRecurrent):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFAdExRefracParameters = LIFAdExRefracParameters(),
        **kwargs,
    ):
        super().__init__(
            activation=lif_adex_refrac_step,
            state_fallback=self.initial_state,
            input_size=input_size,
            hidden_size=hidden_size,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFAdExRefracState:
        dims = (
            *input_tensor.shape[1:],
            self.hidden_size,
        )

        state = LIFAdExRefracState(
            LIFAdExState(
            z=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            v=torch.full(
                dims,
                self.p.v_leak.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            i=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            a=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            rho=torch.zeros(*dims, device=input_tensor.device, dtype=input_tensor.dtype)
            )
        )
        state.lif_adex.v.requires_grad = True
        return state
