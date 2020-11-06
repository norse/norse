"""
This module describes synapse dynamics meant to be injected between
neuron activation modules.
"""

from abc import abstractmethod
import torch
from torch.nn.parameter import Parameter
from typing import Callable

from norse.torch.functional.stdp import STDPParameters, STDPState, stdp_step_linear


class _SynapseBase(torch.nn.Module):
    """
    A synapse models connectivity between two neuron populations.
    Specifically, it keeps track of any state necessary to update the connectivity
    weights between the source (pre) and target (post) populations.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        weights: torch.Tensor = None,
        transformation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        """
        Constructs a generic synapse that takes a input spikes from a previous module with ``input_features`` neurons to an
        output module expecting ``output_features`` neuron inputs. The transformation occurs with a function taking inputs and synapse weights
        as its input and producing output spikes as its output. The function defaults to :meth:`torch.nn.functional.linear`.

        Parameters:
            input_features (int): Number of input neurons from which spikes are being received
            output_features (int): Number of output neurons to which spikes are being sent
            weights (torch.Tensor): Weights to apply to the input signals. Defaults to `kaiming uniform <https://pytorch.org/docs/stable/nn.init.html?highlight=kaiming_uniform#torch.nn.init.kaiming_uniform_>`_.
                                    Expected shape (output_features, input_features)
            transformation (Callable): A function that takes inputs and weights to produce output spikes
        """
        self.input_features = input_features
        self.output_features = output_features
        if weights is None:
            self.weights = Parameter(torch.Tensor(output_features, input_features))
            self.reset_parameters()
        else:
            self.weights = Parameter(weights)
        self.transformation = (
            transformation if transformation is not None else torch.nn.functional.linear
        )

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform(self.weights)

    def forward(self, input_spikes):
        self.register_buffer("presynaptic", input_spikes)
        return self.transformation(input_spikes, self.weights)

    @abstractmethod
    def update(self, postsynaptic):
        """
        Calculates and updates the synapse weights given the spikes from the output layer.
        """
        raise NotImplementedError()


class STDPSynapse(_SynapseBase):
    """
    Spike-time dependent plasticity (STDP) is a local plasticity algorithm
    that modifies connectivity depending on the timing difference between
    pre- and postsynaptic spike times.

    `https://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity`_.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        weights: torch.tensor = None,
        p: STDPParameters = STDPParameters(),
        stepper=stdp_step_linear,  # TODO: Needs simpler interface
    ):
        super(STDPSynapse, self).__init__(input_features, output_features, weights)
        self.p = p
        self.stepper = stepper

    def update(self, postsynaptic, state: STDPState = None) -> STDPState:
        assert (
            self.presynaptic is not None
        ), "No presynaptic input stored, which indicates that the model was not run forward. \
            Did you either delete the presynaptic buffer or run the layer before updating?"
        if state is None:
            state = STDPState(
                torch.zeros_like(self.presynaptic), torch.zeros_like(postsynaptic)
            )
        new_weights, state = self.stepper(
            self.presynaptic, postsynaptic, self.weights, state
        )
        self.weights = new_weights
        return state
