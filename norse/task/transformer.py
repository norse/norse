"""
Sequence-to-Sequence Modeling with nn.Transformer and TorchText
===============================================================

This is a tutorial on how to train a sequence-to-sequence model
that uses the
`nn.Transformer <https://pytorch.org/docs/master/nn.html?highlight=nn%20transformer#torch.nn.Transformer>`__ module.

PyTorch 1.2 release includes a standard transformer module based on the
paper `Attention is All You
Need <https://arxiv.org/pdf/1706.03762.pdf>`__. The transformer model
has been proved to be superior in quality for many sequence-to-sequence
problems while being more parallelizable. The ``nn.Transformer`` module
relies entirely on an attention mechanism (another module recently
implemented as `nn.MultiheadAttention <https://pytorch.org/docs/master/nn.html?highlight=multiheadattention#torch.nn.MultiheadAttention>`__) to draw global dependencies
between input and output. The ``nn.Transformer`` module is now highly
modularized such that a single component (like `nn.TransformerEncoder <https://pytorch.org/docs/master/nn.html?highlight=nn%20transformerencoder#torch.nn.TransformerEncoder>`__
in this tutorial) can be easily adapted/composed.

.. image:: ../_static/img/transformer_architecture.jpg

This example is a modified version of https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py
which is licensed under the BSD 3-Clause License
"""

######################################################################
# Define the model
# ----------------
#


######################################################################
# In this tutorial, we train ``nn.TransformerEncoder`` model on a
# language modeling task. The language modeling task is to assign a
# probability for the likelihood of a given word (or a sequence of words)
# to follow a sequence of words. A sequence of tokens are passed to the embedding
# layer first, followed by a positional encoding layer to account for the order
# of the word (see the next paragraph for more details). The
# ``nn.TransformerEncoder`` consists of multiple layers of
# `nn.TransformerEncoderLayer <https://pytorch.org/docs/master/nn.html?highlight=transformerencoderlayer#torch.nn.TransformerEncoderLayer>`__. Along with the input sequence, a square
# attention mask is required because the self-attention layers in
# ``nn.TransformerEncoder`` are only allowed to attend the earlier positions in
# the sequence. For the language modeling task, any tokens on the future
# positions should be masked. To have the actual words, the output
# of ``nn.TransformerEncoder`` model is sent to the final Linear
# layer, which is followed by a log-Softmax function.
#

from argparse import ArgumentParser
import math

import torch
import torch.nn as nn

from torch.nn import TransformerEncoder
from norse.torch.models.transformer import SpikingTransformerEncoderLayer


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = SpikingTransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


######################################################################
# Load and batch data
# -------------------
#


######################################################################
# The training process uses Wikitext-2 dataset from ``torchtext``. The
# vocab object is built based on the train dataset and is used to numericalize
# tokens into tensors. Starting from sequential data, the ``batchify()``
# function arranges the dataset into columns, trimming off any tokens remaining
# after the data has been divided into batches of size ``batch_size``.
# For instance, with the alphabet as the sequence (total length of 26)
# and a batch size of 4, we would divide the alphabet into 4 sequences of
# length 6:
#
# .. math::
#   \begin{bmatrix}
#   \text{A} & \text{B} & \text{C} & \ldots & \text{X} & \text{Y} & \text{Z}
#   \end{bmatrix}
#   \Rightarrow
#   \begin{bmatrix}
#   \begin{bmatrix}\text{A} \\ \text{B} \\ \text{C} \\ \text{D} \\ \text{E} \\ \text{F}\end{bmatrix} &
#   \begin{bmatrix}\text{G} \\ \text{H} \\ \text{I} \\ \text{J} \\ \text{K} \\ \text{L}\end{bmatrix} &
#   \begin{bmatrix}\text{M} \\ \text{N} \\ \text{O} \\ \text{P} \\ \text{Q} \\ \text{R}\end{bmatrix} &
#   \begin{bmatrix}\text{S} \\ \text{T} \\ \text{U} \\ \text{V} \\ \text{W} \\ \text{X}\end{bmatrix}
#   \end{bmatrix}
#
# These columns are treated as independent by the model, which means that
# the dependence of ``G`` and ``F`` can not be learned, but allows more
# efficient batch processing.
#

import torchtext
from torchtext.data.utils import get_tokenizer


def load_data(args):
    TEXT = torchtext.data.Field(
        tokenize=get_tokenizer("basic_english"),
        init_token="<sos>",
        eos_token="<eos>",
        lower=True,
    )
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)

    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        (train_txt, val_txt, test_txt), batch_size=args.batch_size, bptt_len=args.bptt
    )
    return train_iter, val_iter, test_iter, len(TEXT.vocab.stoi)
    # def batchify(data, bsz):
    #     data = TEXT.numericalize([data.examples[0].text])
    #     # Divide the dataset into bsz parts.
    #     nbatch = data.size(0) // bsz
    #     # Trim off any extra elements that wouldn't cleanly fit (remainders).
    #     data = data.narrow(0, 0, nbatch * bsz)
    #     # Evenly divide the data across the bsz batches.
    #     data = data.view(bsz, -1).t().contiguous()
    #     return data

    # train_data = batchify(train_txt, args.batch_size)
    # val_data = batchify(val_txt, args.batch_size)
    # test_data = batchify(test_txt, args.batch_size)
    # return train_data, val_data, test_data, len(TEXT.vocab.stoi)


######################################################################
# Initiate PyTorch Lightning training
# ------------------------------------
#
import pytorch_lightning as pl


class PLModel(pl.LightningModule):

    ######################################################################
    # The model is set up with the hyperparameter below. The vocab size is
    # equal to the length of the vocab object.
    #
    def __init__(self, ntokens, bptt):
        super().__init__()
        self.ntokens = ntokens  # the size of vocabulary
        self.emsize = 200  # embedding dimension
        self.nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
        self.nlayers = (
            2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        )
        self.nhead = 2  # the number of heads in the multiheadattention models
        self.dropout = 0.2  # the dropout value
        self.bptt = bptt
        self.loss_function = nn.CrossEntropyLoss()
        self.model = TransformerModel(
            self.ntokens, self.emsize, self.nhead, self.nhid, self.nlayers, self.dropout
        )

    def forward(self, x):
        return self.model(x)

    ######################################################################
    # We define a regular SGD optimiser with stepped learning rates.
    #

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        return optimizer

    ######################################################################
    # `CrossEntropyLoss <https://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss>`__
    # is applied to track the loss and
    # `SGD <https://pytorch.org/docs/master/optim.html?highlight=sgd#torch.optim.SGD>`__
    # implements stochastic gradient descent method as the optimizer. The initial
    # learning rate is set to 5.0. `StepLR <https://pytorch.org/docs/master/optim.html?highlight=steplr#torch.optim.lr_scheduler.StepLR>`__ is
    # applied to adjust the learn rate through epochs. During the
    # training, we use
    # `nn.utils.clip_grad_norm\_ <https://pytorch.org/docs/master/nn.html?highlight=nn%20utils%20clip_grad_norm#torch.nn.utils.clip_grad_norm_>`__
    # function to scale all the gradient together to prevent exploding.
    #

    def training_step(self, batch, batch_idx):
        ntokens = self.ntokens
        src_mask = self.model.generate_square_subsequent_mask(self.bptt).to(self.device)
        data, targets = batch.text, batch.target
        output = self.model(data, src_mask)
        loss = self.loss_function(output.view(-1, ntokens), targets.view(-1))
        self.scheduler.step()  # Step the scheduler manually
        return loss

    def test_step(self, batch, batch_idx):
        ntokens = self.ntokens
        src_mask = self.model.generate_square_subsequent_mask(self.bptt).to(self.device)
        with torch.no_grad():
            data, targets = batch.text, batch.target
            output = self.model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            return len(data) * self.loss_function(output_flat, targets.view(-1)).item()


######################################################################
# Run the model
# -------------
#


def main(args):

    ######################################################################
    # Train the model via PyTorch Lightning

    data_train, data_val, data_test, ntokens = load_data(args)
    trainer = pl.Trainer.from_argparse_args(args)
    model = PLModel(ntokens, args.bptt)
    trainer.fit(model, data_train, data_val)

    ######################################################################
    # Evaluate the model with the test dataset
    # -------------------------------------
    #
    # Apply the best model to check the result with the test dataset.

    test_loss = trainer.test()
    print("=" * 89)
    print(
        "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
            test_loss, math.exp(test_loss)
        )
    )
    print("=" * 89)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--bptt", default=35)
    args = parser.parse_args()
    args.device = "cuda" if "gpus" in args else "cpu"
    main(args)
