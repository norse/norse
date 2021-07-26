import torch
import torch.jit
from typing import NamedTuple

import MinkowskiEngine as ME

from norse.torch import LIFFeedForwardState, LIFCell


class MinkowskiLIFState(NamedTuple):
    v: torch.Tensor
    i: torch.Tensor
    z: torch.Tensor


class MinkowskiLIFCell(ME.MinkowskiModuleBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.module = LIFCell(*args, **kwargs)
        self.pruning = ME.MinkowskiPruning()
        self.union = ME.MinkowskiUnion()

    def forward(self, input, state):
        if state is not None:
            # Align state with input
            zero_state_tensor = ME.SparseTensor(
                features=torch.zeros_like(state.v.F),
                coordinates=state.v.coordinates,
                coordinate_manager=input.coordinate_manager,
            )
            v_tensor = self.union(
                input,
                ME.SparseTensor(
                    features=state.v.F,
                    coordinates=state.v.C,
                    coordinate_manager=input.coordinate_manager,
                ),
            )
            i_tensor = self.union(
                input,
                ME.SparseTensor(
                    features=state.i.F,
                    coordinates=state.i.C,
                    coordinate_manager=input.coordinate_manager,
                ),
            )
            state = LIFFeedForwardState(v_tensor.F, i_tensor.F)
            input = self.union(input, zero_state_tensor)
        else:
            zero_input_tensor = ME.SparseTensor(
                features=torch.zeros_like(input.F),
                coordinates=input.coordinates,
                coordinate_manager=input.coordinate_manager,
            )
            v_tensor, i_tensor = zero_input_tensor, zero_input_tensor
        output, s_out = self.module(input.F, state)
        return (
            ME.SparseTensor(
                features=output,
                coordinates=v_tensor.C,
                coordinate_manager=input.coordinate_manager,
            ),
            MinkowskiLIFState(
                v=ME.SparseTensor(
                    features=s_out.v,
                    coordinates=v_tensor.C,
                    coordinate_manager=input.coordinate_manager,
                ),
                i=ME.SparseTensor(
                    features=s_out.i,
                    coordinates=i_tensor.C,
                    coordinate_manager=input.coordinate_manager,
                ),
                z=output,
            ),
        )
