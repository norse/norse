import torch
import torch.jit
import MinkowskiEngine as ME

from norse.torch.functional.lif import (
    LIFState,
    LIFFeedForwardState,
    LIFParameters,
    lif_step,
    lif_feed_forward_step,
)
from norse.torch.module.lif import (
    LIFCell,
    LIFRecurrentCell
)
from typing import NamedTuple, Tuple



class MinkowskiLIFCell(ME.MinkowskiModuleBase):
    def __init__(self, *args, **kwargs):
        super(MinkowskiLIFCell, self).__init__()
        self.module = LIFCell(*args, **kwargs)

    def forward(self, input, state):
        s = LIFFeedForwardState(state.v.F, state.i.F)
        output, s_out = self.module(input.F, s)
        if isinstance(input, ME.TensorField):
            return ME.TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            ), LIFFeedForwardState(
                ME.TensorField(
                    s_out.v,
                    coordinate_field_map_key=state.v.coordinate_field_map_key,
                    coordinate_manager=state.v.coordinate_manager,
                    quantization_mode=state.v.quantization_mode,
                ),
                ME.TensorField(
                    s_out.i,
                    coordinate_field_map_key=state.i.coordinate_field_map_key,
                    coordinate_manager=state.i.coordinate_manager,
                    quantization_mode=state.i.quantization_mode,
                ),
            )
        else:
            return ME.SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            ), LIFFeedForwardState(
                ME.SparseTensor(
                    s_out.v,
                    coordinate_field_map_key=state.v.coordinate_field_map_key,
                    coordinate_manager=state.v.coordinate_manager,
                    quantization_mode=state.v.quantization_mode,
                ),
                ME.SparseTensor(
                    s_out.i,
                    coordinate_field_map_key=state.i.coordinate_field_map_key,
                    coordinate_manager=state.i.coordinate_manager,
                    quantization_mode=state.i.quantization_mode,
                ),
            )

    def __repr__(self):
        return self.__class__.__name__ + "()"