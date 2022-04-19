import torch
import warnings
from typing import Any, Callable, Optional

from norse.torch.functional.lif import LIFFeedForwardState
from norse.torch.module.snn import SNN, FeedforwardActivation

# ===================================================================
# Implement various SNN cells using CUDA Graphs
# <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>
# ===================================================================


class SNNCellCG(SNN):
    """
    Initializes a feedforward neuron cell *without* time.
    Accelerated forward pass using Pytorch with CUDA Graphs.
    <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>

    Parameters:
        activation (FeedforwardActivation): The activation function accepting an input tensor, state
            tensor, and parameters module, and returning a tuple of (output spikes, state).
        state_fallback (Callable[[torch.Tensor], Any]): A function that can return a
            default state with the correct dimensions, in case no state is provided in the
            forward pass.
        p (torch.nn.Module): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        dt (float): Time step to use in integration. Defaults to 0.001.
        activation_sparse (Optional[FeedforwardActivation]): A Sparse activation function - if it exists
            for the neuron model
    """

    def __init__(
        self,
        activation: FeedforwardActivation,
        state_fallback: Callable[[torch.Tensor], torch.Tensor],
        p: Any,
        dt: float = 0.001,
        activation_sparse: Optional[FeedforwardActivation] = None,
    ):
        super(SNNCellCG, self).__init__(
            activation=activation,
            state_fallback=state_fallback,
            p=p,
            dt=dt,
            activation_sparse=activation_sparse,
        )
        # warn when using experimental CG cell
        warnings.warn(
            "{} is an experimental feature and could change at any time.".format(
                self.__class__
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )

        # configure internal CUDAGraph
        self.config = True

        # graphed callable
        self.activation_cg = None

    def forward(self, input_tensor: torch.Tensor, state: Optional[Any] = None):
        state = state if state is not None else self.state_fallback(input_tensor)

        if self.config:
            # capture input
            sample_args = (
                torch.randn(
                    input_tensor.shape, device=input_tensor.device, requires_grad=True
                ),
            )  # account for input tensor
            for state_variable in state:
                sample_args += (
                    torch.randn(
                        state_variable.shape,
                        device=state_variable.device,
                        requires_grad=True,
                    ),
                )  # account for state

            if self.activation_sparse is not None and input_tensor.is_sparse:
                raise NotImplementedError(
                    "Cuda Graph version for sparse activation not yet implemented!"
                )
            else:
                self.activation_cg = torch.cuda.make_graphed_callables(
                    self.activation, sample_args
                )

            self.config = False

        # collect inputs
        sample_args = (input_tensor,)
        for state_variable in state:
            sample_args += (state_variable,)

        # run forward pass
        out_tuple = self.activation_cg(*sample_args)

        if isinstance(state, LIFFeedForwardState):
            out_state = LIFFeedForwardState(v=out_tuple[1], i=out_tuple[2])
        else:
            raise NotImplementedError(
                "Cuda Graph version for SNNCell other than LIFCell not yet implemented!"
            )

        return out_tuple[0], out_state
