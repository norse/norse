from typing import Union

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
        >>> data = torch.ones(1, 16, 8, 4)         # Single timestep
        >>> model = snn.SequentialState(
        >>>   snn.Lift(torch.nn.Conv2d(16, 8, 3)), # (1, 8, 6, 2)
        >>>   torch.nn.Flatten(2),                 # (1, 8, 12)
        >>>   snn.LIFRecurrent(12, 6),             # (1, 8, 6)
        >>>   snn.LIFRecurrent(6, 1)               # (1, 8, 1)
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
        self.output_handles = []
        self.spike_history = []
        self.state_history = []
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
            # Identify all the stateful layers
            signature = inspect.signature(module.forward)
            self.stateful_layers.append(
                "state" in signature.parameters or isinstance(module, torch.nn.RNNBase)
            )

    def register_debug_hooks(self):
        """
        Registers debug hooks that captures spikes (:attr:`output_spikes`) and states
        (:attr:`output_states`) for every stateful layer.

        Hooks can be removed by calling :meth:`remove_debug_hook`_.

        Example:
            >>> import norse.torch as snn
            >>> module = snn.SequentialState(...)
            >>> module.register_debug_hook()
            >>> module(...)
            >>> module.output_spikes # Layer spikes from last application
            >>> module.output_states # Layer states from last application
        """
        if len(self.output_handles) > 0:
            raise ValueError("Debug hooks already in place")

        def output_handle(mod, inp, out):
            self.spike_history.append(out[0])
            self.state_history.append(out[1])

        def clear_output(mod, inp):
            self.spike_history = []
            self.state_history = []

        for name, module in self.named_children():
            if self.stateful_layers[int(name)]:
                handle = module.register_forward_hook(output_handle)
                self.output_handles.append(handle)
        self.output_handles.append(self.register_forward_pre_hook(clear_output))

    def remove_debug_hooks(self):
        """
        Disables the debug hooks, registered in :meth:`register_debug_hook`_.
        """
        for handle in self.output_handles:
            handle.remove()
        del self.output_spikes
        del self.output_states
        del self.output_handles

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
