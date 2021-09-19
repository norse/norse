from typing import Callable, Union

import inspect
import torch


class SequentialState(torch.nn.Sequential):
    """
    A sequential model that works like PyTorch's ``Sequential`` with the
    addition that it handles neuron states.


    Arguments:
      args (*torch.nn.Module): A list of modules to sequentially apply in the forward pass

    Example:
        >>> import torch
        >>> import norse.torch as snn
        >>> data = torch.ones(1, 1, 16, 8, 4)         # Single timestep, Single Batch, 16 channels
        >>> model = snn.SequentialState(
        >>>   snn.Lift(torch.nn.Conv2d(16, 8, 3)), # (1, 1, 8, 6, 2)
        >>>   torch.nn.Flatten(3),                 # (1, 1, 8, 12)
        >>>   snn.LIFRecurrent(12, 6),             # (1, 1, 8, 6)
        >>>   snn.LIFRecurrent(6, 1)               # (1, 1, 8, 1)
        >>> )
        >>> model(data)

    Example with recurrent layers:
        >>> import torch
        >>> import norse.torch as snn
        >>> data = torch.ones(1, 16, 8, 4)         # Single timestep
        >>> model = snn.SequentialState(
        >>>   snn.Lift(torch.nn.Conv2d(16, 8, 3)), # (1, 8, 6, 2)
        >>>   torch.nn.Flatten(2),                 # (1, 8, 12)
        >>>   snn.LSNNRecurrent(12, 6),            # (1, 8, 6)
        >>>   torch.nn.RNN(6, 4, 2),               # (1, 6, 4) with 2 recurrent layers
        >>>   snn.LIFRecurrent(4, 1)               # (1, 4, 1)
        >>> )
        >>> model(data)
    """

    def __init__(self, *args: torch.nn.Module):
        super(SequentialState, self).__init__()
        self.stateful_layers = []
        self.forward_state_hooks = []
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
            # Identify all the stateful layers
            signature = inspect.signature(module.forward)
            self.stateful_layers.append(
                "state" in signature.parameters or isinstance(module, torch.nn.RNNBase)
            )

    def register_forward_state_hooks(
        self,
        forward_hook: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None],
    ):
        """
        Registers hooks for all state*ful* layers.

        Hooks can be removed by calling :meth:`remove_state_hooks`_.

        Arguments:
          child_hook (Callable): The hook applied to all children everytime they produce an output
          pre_hook (Optional[Callable]): An optional hook for the SequentialState module,
                                         executed *before* the input is propagated to the children.

        Example:
            >>> import norse.torch as snn
            >>> def my_hook(module, input, output):
            >>>     ...
            >>> module = snn.SequentialState(...)
            >>> module.register_forward_state_hook(my_hook)
            >>> module(...)
        """
        if len(self.forward_state_hooks) > 0:
            raise ValueError("Forward state hooks already in place")

        for name, module in self.named_children():
            if self.stateful_layers[int(name)]:
                handle = module.register_forward_hook(forward_hook)
                self.forward_state_hooks.append(handle)

    def remove_forward_state_hooks(self):
        """
        Disables and discards the forward state hooks, registered in :meth:`register_forward_state_hooks`_.
        """
        for handle in self.forward_state_hooks:
            handle.remove()
        self.forward_state_hooks.clear()

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
