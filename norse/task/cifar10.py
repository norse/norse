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

from norse.torch import LIFParameters, LIFFeedForwardCell, LIFeedForwardCell
from norse.torch import ConvNet, ConvNet4
from norse.torch import (
    ConstantCurrentLIFEncoder,
    PoissonEncoder,
    SignedPoissonEncoder,
    SpikeLatencyLIFEncoder,
)
from norse.torch import Lift, SequentialState, RegularizationWrapper


class LIFConvNet(pl.LightningModule):
    def __init__(self, num_channels, lr, optimizer, p):
        super(LIFConvNet, self).__init__()
        self.lr = lr
        self.optimizer = optimizer
        self.rsnn = SequentialState(
            torch.nn.Conv2d(num_channels, 128, 3),  # 0
            RegularizationWrapper(LIFFeedForwardCell(p)),
            torch.nn.Conv2d(128, 256, 3),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.BatchNorm2d(256),
            RegularizationWrapper(LIFFeedForwardCell(p)),  # 5
            torch.nn.Conv2d(256, 512, 3),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(512, 1024, 3),
            torch.nn.Conv2d(1024, 512, 3),  # 10
            torch.nn.Flatten(1),
            torch.nn.Linear(2048, 1024),
            RegularizationWrapper(LIFFeedForwardCell(p)),
            torch.nn.Linear(1024, 512),
            RegularizationWrapper(LIFFeedForwardCell(p)),  # 15
            torch.nn.Linear(512, 10),
            LIFeedForwardCell(),
        )

    def forward(self, x):
        voltages = torch.zeros(*x.shape[:2], 10, device=x.device, dtype=x.dtype)
        s = None
        for ts in range(x.shape[0]):
            out, s = self.rsnn(x[ts], s)  # .permute(1, 0, 2)
            # m, _ = torch.max(voltages, 0)
            voltages[ts, :, :] = out

        m, _ = voltages.max(1)
        regularization = (s[1].count + s[5].count + s[13].count + s[15].count) * 1e-6
        return torch.nn.functional.log_softmax(m, dim=1), regularization

    def training_step(self, batch, batch_idx):
        x, y = batch
        out, regularization = self(x)
        loss = torch.nn.functional.nll_loss(out, y) + regularization
        return loss

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            return torch.optim.SGD(self.parameters(), lr=self.lr)


def main(args):

    # Set seeds
    torch.manual_seed(args.manual_seed)

    # Setup encoding
    num_channels = 4
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
    def add_luminance(images):
        return torch.cat(
            (
                images,
                torch.unsqueeze(
                    0.2126 * images[0, :, :]
                    + 0.7152 * images[1, :, :]
                    + 0.0722 * images[2, :, :],
                    0,
                ),
            ),
            0,
        )

    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]
        + [add_luminance, encoder]
    )
    transform_test = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()] + [add_luminance, encoder]
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
        num_channels=num_channels, lr=args.lr, optimizer=args.optimizer, p=p
    )
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)
    trainer.test(test_dataloader=test_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=1000, auto_select_gpus=True, progress_bar_refresh_rate=1
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Number of examples in one minibatch"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate to use.")
    parser.add_argument(
        "--current_encoder_v_th",
        type=float,
        default=0.7,
        help="Voltage threshold for the LIF dynamics",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="constant",
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
        "--seq_length", default=100, type=int, help="Number of timesteps to do."
    )
    parser.add_argument(
        "--manual_seed", default=0, type=int, help="Random seed for torch"
    )
    args = parser.parse_args()

    main(args)
