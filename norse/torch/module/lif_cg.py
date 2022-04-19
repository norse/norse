import torch

from norse.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParameters,
    lif_feed_forward_step_sparse,
)

from norse.torch.functional.adjoint.lif_adjoint import (
    lif_feed_forward_adjoint_step,
    lif_feed_forward_adjoint_step_sparse,
)

from norse.torch.functional.lif_cg import (
    lif_cg_feed_forward_step,
)

from norse.torch.module.snn_cg import SNNCellCG


class LIFCellCG(SNNCellCG):
    """Module that computes a single euler-integration step of a
    leaky integrate-and-fire (LIF) neuron-model *without* recurrence and *without* time.

    Accelerated forward pass using Pytorch with CUDA Graphs.
    <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>

    More specifically it implements one integration step
    of the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}})

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}}
        \\end{align*}

    Example:
        >>> data = torch.zeros(5, 2) # 5 batches, 2 neurons
        >>> l = LIFCell(2, 4)
        >>> l(data) # Returns tuple of (Tensor(5, 4), LIFState)

    Arguments:
        p (LIFParameters): Parameters of the LIF neuron model.
        sparse (bool): Whether to apply sparse activation functions (True) or not (False). Defaults to False.
        dt (float): Time step to use. Defaults to 0.001.
    """

    def __init__(self, p: LIFParameters = LIFParameters(), **kwargs):
        super().__init__(
            activation=(
                lif_feed_forward_adjoint_step
                if p.method == "adjoint"
                else lif_cg_feed_forward_step
            ),
            activation_sparse=lif_feed_forward_adjoint_step_sparse
            if p.method == "adjoint"
            else lif_feed_forward_step_sparse,
            state_fallback=self.initial_state,
            p=LIFParameters(
                torch.as_tensor(p.tau_syn_inv),
                torch.as_tensor(p.tau_mem_inv),
                torch.as_tensor(p.v_leak),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFFeedForwardState:
        state = LIFFeedForwardState(
            v=torch.full(
                input_tensor.shape,
                torch.as_tensor(self.p.v_leak).detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                *input_tensor.shape,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state
