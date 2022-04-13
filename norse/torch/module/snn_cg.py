import torch
from typing import Any, Callable, List, Optional, Tuple

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
            activation_sparse=activation_sparse
        )
        # create internal CUDAGraph
        self.config = True 

        # graphed callable
        self.activation_cg = None


    def add(self, x): 
        return x + 20


    def forward(self, input_tensor: torch.Tensor, state: Optional[Any] = None):
        state = state if state is not None else self.state_fallback(input_tensor)

        if self.config:
            # capture input
            sample_args = (torch.randn(input_tensor.shape, device=input_tensor.device, requires_grad=True),) # account for input tensor
            for state_variable in state: 
                sample_args += (torch.randn(state_variable.shape, device=state_variable.device, requires_grad=True),)

            if self.activation_sparse is not None and input_tensor.is_sparse:
                self.activation_cg = torch.cuda.make_graphed_callables(self.activation_sparse, sample_args)
            else:
                self.activation_cg = torch.cuda.make_graphed_callables(self.activation, sample_args)

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
            raise NotImplementedError("Cuda Graph version for SNNCell other than LIFCell not yet implemented!")

        return out_tuple[0], out_state

'''
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
            activation_sparse=activation_sparse
        )
        # warm up workload to be captured
        self.warmup = True

        # placeholders for capture
        self.static_input_tensor = None
        self.static_state = None
        self.z = None
        self.s = None 

        # stream
        self.g = None


    def forward(self, input_tensor: torch.Tensor, state: Optional[Any] = None):
        state = state if state is not None else self.state_fallback(input_tensor)

        if self.warmup:
            # define tensors that hold input during capture
            self.static_input_tensor = torch.randn(input_tensor.shape).type_as(input_tensor) 
            self.static_state = state
            #for entry in self.static_state:
            #    entry.requires_grad = False

            # warm up cuda stream
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(stream):
                if self.activation_sparse is not None and input_tensor.is_sparse:
                    for step in range(3):
                        _ = self.activation_sparse(self.static_input_tensor, self.static_state, self.p, self.dt)
                else:
                    for step in range(3):
                        _ = self.activation(self.static_input_tensor, self.static_state, self.p, self.dt)
            torch.cuda.current_stream().wait_stream(stream)
        
            # capture cuda ops
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                if self.activation_sparse is not None and input_tensor.is_sparse:
                    self.z, self.s = self.activation_sparse(self.static_input_tensor, self.static_state, self.p, self.dt)
                    self.static_state.v.requires_grad = False
                else:
                    self.z, self.s = self.activation(self.static_input_tensor, self.static_state, self.p, self.dt)
                    self.static_state.v.requires_grad = False

            # end warmup
            self.warmup = False

        # copy input to placeholders
        self.static_input_tensor.copy_(input_tensor)
        if isinstance(state, LIFFeedForwardState):
            self.static_state.v.copy_(state.v)
            self.static_state.i.copy_(state.i)
            print(f"static state: {self.static_state}")
            #for i, entry in enumerate(state):
            #    self.static_state[i].copy_(entry)
            #    if i == 0: 
            #       self.static_state[i].requires_grad = True
        #else: 
        #    raise NotImplemented("Feedforward states other than 'LIFFeedForwardState' are not yet implemented!")

        # replay
        self.g.replay()

        #print(f"z: {self.z}")
        #print(f"s: {state.v}")

        return self.z, self.s
'''