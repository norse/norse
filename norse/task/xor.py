"""
This is a rework of the TorchDYN neural ODE problem presented here: 
https://towardsdatascience.com/neural-odes-with-pytorch-lightning-and-torchdyn-87ca4a7c6ffd
"""
from argparse import ArgumentParser

import torch
import torch.utils.data as data

import torch.nn as nn
import pytorch_lightning as pl

import norse.torch as norse


class XORModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = norse.SequentialState(
            nn.Linear(2, 2),
            norse.LIFFeedForwardCell(),
            nn.Linear(2, 1),
            norse.LIFFeedForwardCell(),
        )

    def forward(self, x):
        state = None
        for i in range(10):  # Simulate for some timesteps
            out, state = self.model(x, state)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = nn.BCELoss()(x_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-6)


class XORDataset(data.Dataset):
    def __init__(self, n=1000):
        self.x = torch.FloatTensor([[0, 0], [1, 0], [0, 1], [1, 1]])
        self.y = torch.FloatTensor([[0], [1], [1], [0]])
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (self.x[idx % len(self.x)], self.y[idx % len(self.y)])


def main(args):
    train = XORDataset(1000)
    trainloader = data.DataLoader(train, batch_size=16, shuffle=True)
    trainer = pl.Trainer.from_argparse_args(args)
    model = XORModel()
    trainer.fit(model, trainloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=1000, auto_select_gpus=True, progress_bar_refresh_rate=1
    )
    args = parser.parse_args()

    main(args)
