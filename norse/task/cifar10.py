import os
import datetime
import uuid

from argparse import ArgumentParser
from collections import namedtuple
import torchvision
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import pytorch_lightning as pl

from norse.torch import LIFParameters, LIFCell, LICell, LIParameters
from norse.torch import (
    ConstantCurrentLIFEncoder,
    PoissonEncoder,
    SignedPoissonEncoder,
    SpikeLatencyLIFEncoder,
)
from norse.torch import SequentialState, RegularizationWrapper


class LIFConvNet(pl.LightningModule):
    def __init__(self, seq_length, num_channels, lr, optimizer, p, lr_step=True):
        super().__init__()
        self.lr = lr
        self.lr_step = lr_step
        self.optimizer = optimizer
        self.seq_length = seq_length

        self.rsnn = SequentialState(
            torch.nn.Conv2d(num_channels, 128, 3),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 4),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 256, 6),
            torch.nn.Flatten(1),
            torch.nn.Linear(256, 256),
            RegularizationWrapper(LIFCell(p)),
            torch.nn.Linear(256, 128),
            RegularizationWrapper(LIFCell(p)),
            torch.nn.Linear(128, 10),
            LICell(),
        )

    def forward(self, x):
        # X was shape (batch, time, ...) and will be (time, batch, ...)
        x = x.permute(1, 0, 2, 3, 4)
        voltages = torch.empty(*x.shape[:2], 10, device=x.device, dtype=x.dtype)
        s = None
        for ts in range(x.shape[0]):
            out, s = self.rsnn(x[ts], s)
            voltages[ts, :, :] = out

        # Regularize all spiking layers to a number of spikes within 1% - 20%
        regularization = 0
        for substate in s:
            if hasattr(substate, "count") and isinstance(substate.count, torch.Tensor):
                min_spikes = substate.state.v.shape[-1] * self.seq_length * 0.01  # 1%
                max_spikes = min_spikes * 20  # 20%
                regularization = regularization + max(0, min_spikes - substate.count)
                regularization = regularization + max(0, substate.count - max_spikes)

        return voltages, regularization

    # Forward pass of a single batch
    def training_step(self, batch, batch_idx):
        x, y = batch
        out, r = self(x)
        loss = torch.nn.functional.cross_entropy(out.mean(0), y) + r

        self.log("Reg.", r, prog_bar=True)
        self.log("Loss", loss)
        self.log("LR", self.scheduler.get_last_lr()[0])
        return loss

    # Same as in training_step, but also reports accuracy
    def test_step(self, batch, batch_idx):
        x, y = batch
        out, _ = self(x)
        loss = torch.nn.functional.cross_entropy(out.mean(0), y)
        classes = out.mean(0).argmax(1)
        acc = torch.eq(classes, y).sum().item() / len(y)

        self.log("Loss", loss)
        self.log("Acc.", acc)
        self.log("LR", self.scheduler.get_last_lr()[0])

    def training_epoch_end(self, outputs):
        if self.lr_step:
            self.scheduler.step()

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=1e-5
            )
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
        return optimizer


def main(args):

    # Set seeds
    torch.manual_seed(args.manual_seed)

    # Setup encoding
    num_channels = 3
    p = LIFParameters(v_th=torch.as_tensor(args.current_encoder_v_th))
    constant_encoder = ConstantCurrentLIFEncoder(seq_length=args.seq_length, p=p)
    if args.encoding == "poisson":
        encoder = PoissonEncoder(seq_length=args.seq_length, f_max=200)
    elif args.encoding == "constant":
        encoder = constant_encoder
    elif args.encoding == "constant_first":
        encoder = SpikeLatencyLIFEncoder(seq_length=args.seq_length, p=p)
    elif args.encoding == "signed_poisson":
        encoder = SignedPoissonEncoder(seq_length=args.seq_length, f_max=200)
    elif args.encoding == "signed_constant":

        def signed_current_encoder(x):
            z = constant_encoder(torch.abs(x))
            return torch.sign(x) * z

        encoder = signed_current_encoder
    elif args.encoding == "constant_polar":

        def polar_current_encoder(x):
            x_p = constant_encoder(2 * torch.nn.functional.relu(x))
            x_m = constant_encoder(2 * torch.nn.functional.relu(-x))
            return torch.cat((x_p, x_m), 1)

        encoder = polar_current_encoder
        num_channels = 2 * num_channels

    # Load datasets
    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(0.5, 0.5, 0.5)),
            encoder,
        ]
    )
    transform_test = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), encoder]
    )
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root=".", train=True, download=True, transform=transform_train
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root=".", train=False, transform=transform_test),
        batch_size=args.batch_size,
    )

    # Define and train the model
    model = LIFConvNet(
        seq_length=args.seq_length,
        num_channels=num_channels,
        lr=args.lr,
        optimizer=args.optimizer,
        p=p,
    )
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)
    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=1000, auto_select_gpus=True, progress_bar_refresh_rate=1
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Number of examples in one minibatch"
    )
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate to use.")
    parser.add_argument(
        "--lr_step",
        type=bool,
        default=True,
        help="Use a stepper to reduce learning weight.",
    )
    parser.add_argument(
        "--current_encoder_v_th",
        type=float,
        default=0.7,
        help="Voltage threshold for the LIF dynamics",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="constant_polar",
        choices=[
            "poisson",
            "constant",
            "constant_first",
            "constant_polar",
            "signed_poisson",
            "signed_constant",
        ],
        help="How to code from CIFAR image to spikes.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--seq_length", default=40, type=int, help="Number of timesteps to do."
    )
    parser.add_argument(
        "--manual_seed", default=0, type=int, help="Random seed for torch"
    )
    args = parser.parse_args()

    main(args)
