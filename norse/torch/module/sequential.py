import inspect
import torch
from typing import Union


class SequentialState(torch.nn.Sequential):
    """
    A sequential model that works exactly like PyTorch's ``Sequential`` with the
    addition that it handles neuron states.

    Example:
    >>> import torch
    >>> from norse.torch.module.lift import LIFT
    >>> from norse.torch.module.sequential import SequentialState
    >>> from norse.torch.module.lif import LIFLayer
    >>> data = torch.ones(1, 16, 8, 4)     # Single timestep
    >>> model = SequentialState(
    >>>   Lift(torch.nn.Conv2d(16, 8, 3)), # (1, 8, 6, 2)
    >>>   torch.nn.Flatten(2),             # (1, 8, 12)
    >>>   LIFLayer(12, 6),                 # (1, 8, 6)
    >>>   LIFLayer(6, 1)                   # (1, 8, 1)
    >>> )
    >>> model(data)

    Example with recurrent layers:
    >>> import torch
    >>> from norse.torch.module.lift import Lift
    >>> from norse.torch.module.sequential import SequentialState
    >>> from norse.torch.module.lif import LIFCell, LIFLayer
    >>> from norse.torch.module.lsnn import LSNNCell, LSNNLayer
    >>> data = torch.ones(1, 16, 8, 4)     # Single timestep
    >>> model = SequentialState(
    >>>   Lift(torch.nn.Conv2d(16, 8, 3)), # (1, 8, 6, 2)
    >>>   torch.nn.Flatten(2),             # (1, 8, 12)
    >>>   LSNNLayer(LSNNCell, 12, 6),      # (1, 8, 6)
    >>>   torch.nn.RNN(6, 4, 2),           # (1, 6, 4) with 2 recurrent layers
    >>>   LIFLayer(LIFCell,4, 1)           # (1, 4, 1)
    >>> )
    >>> model(data)

    """

    def __init__(self, *args: torch.nn.Module):
        super(SequentialState, self).__init__()
        self.stateful_layers = []
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
            # Identify all the stateful layers
            signature = inspect.signature(module.forward)
            self.stateful_layers.append(
                "state" in signature.parameters or isinstance(module, torch.nn.RNNBase)
            )

    def forward(self, input_tensor: torch.Tensor, state: Union[list, None] = None):
        """
        Feeds the input to the modules with the given state-list.
        If the state is None, the initial state is set to None for each of the modules.

        Parameters:
            input_tensor: The input tensor too feed into the first module
            state: Either a list of states for each module or None. If None, the modules
                   will initialise their own default state

        Returns:
            A tuple of (output tensor, state list)
        """
        state = [None] * len(self) if state is None else state
        for index, module in enumerate(self):
            if self.stateful_layers[index]:
                input_tensor, s = module(input_tensor, state[index])
                state[index] = s
            else:
                input_tensor = module(input_tensor)
        return input_tensor, state
