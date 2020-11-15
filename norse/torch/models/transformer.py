from typing import Optional

import torch
from torch import Tensor
from norse.torch.functional.lif import lif_feed_forward_step
from norse.torch.functional.lift import lift
from norse.torch.module.encode import PoissonEncoder


class SpikingTransformerEncoderLayer(torch.nn.Module):
    """
    A spiking :class:`.TransformerEncoderLayer`_ for the :class:`.Transformer`_ models in PyTorch.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=lif_feed_forward_step,
        seq_length=100,
    ):
        super(SpikingTransformerEncoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.poisson = PoissonEncoder(seq_length, f_max=1000)
        self.activation = lift(activation)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        lin_src, _ = self.activation(self.linear1(self.poisson(src)))
        lin_src = lin_src.mean(0)
        src2 = self.linear2(self.dropout(lin_src))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
