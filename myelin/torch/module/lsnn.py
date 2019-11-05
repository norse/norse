import torch

from ..functional.lsnn import (
    LSNNParameters,
    LSNNState,
    LSNNFeedForwardState,
    lsnn_step,
    lsnn_feed_forward_step,
)

from typing import Tuple, List
import numpy as np


class LSNNCell(torch.nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        p: LSNNParameters = LSNNParameters(),
        dt: float = 0.001,
    ):
        super(LSNNCell, self).__init__()
        self.input_weights = torch.nn.Parameter(
            torch.randn(input_features, output_features) / np.sqrt(input_features)
        )
        self.recurrent_weights = torch.nn.Parameter(
            torch.randn(output_features, output_features)
        )
        self.p = p
        self.dt = dt

    def initial_state(self, batch_size, device, dtype=torch.float) -> LSNNState:
        """return the initial state of an LSNN neuron"""
        return LSNNState(
            z=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            v=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            i=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            b=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
        )

    def forward(
        self, input: torch.Tensor, state: LSNNState
    ) -> Tuple[torch.Tensor, LSNNState]:
        return lsnn_step(
            input,
            state,
            self.input_weights,
            self.recurrent_weights,
            p=self.p,
            dt=self.dt,
        )


class LSNNLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSNNLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(
        self, input: torch.Tensor, state: LSNNState
    ) -> Tuple[torch.Tensor, LSNNState]:
        inputs = input.unbind(0)
        outputs = []  # torch.jit.annotate(List[torch.Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class LSNNFeedForwardCell(torch.nn.Module):
    def __init__(self, shape, p: LSNNParameters = LSNNParameters(), dt: float = 0.001):
        super(LSNNFeedForwardCell, self).__init__()
        self.shape = shape
        self.p = p
        self.dt = dt

    def initial_state(
        self, batch_size, device, dtype=torch.float
    ) -> LSNNFeedForwardState:
        """return the initial state of an LSNN neuron"""
        return LSNNFeedForwardState(
            v=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
            i=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
            b=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
        )

    def forward(
        self, input: torch.Tensor, state: LSNNFeedForwardState
    ) -> Tuple[torch.Tensor, LSNNFeedForwardState]:
        return lsnn_feed_forward_step(input, state, p=self.p, dt=self.dt)
