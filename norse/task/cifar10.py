import os
import datetime
import uuid

from argparse import ArgumentParser
from collections import namedtuple
import torchvision
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

import norse





class LIFConvNet(pl.LightningModule):
    def __init__(
        self, seq_length, num_channels, lr, optimizer, p, noise_scale=1e-6, lr_step=True
    ):
        super().__init__()
        self.lr = lr
        self.lr_step = lr_step
        self.optimizer = optimizer
        self.seq_length = seq_length
        self.p = p
        
        self.features = norse.torch.SequentialState(
            # Convolutional layers
            torch.nn.Conv2d(num_channels, 64, 3),  # Block 1
            norse.torch.LIFCell(p),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3),  # Block 2
            norse.torch.LIFCell(p),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3),  # Block 3
            norse.torch.LIFCell(p),            
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
        )
        self.classification = norse.torch.SequentialState(
            # Classification
            torch.nn.Linear(1024, 10),
            norse.torch.LICell()
        )

    def forward(self, x):
        voltages = torch.empty(self.seq_length, x.shape[0], 10, device=x.device, dtype=x.dtype)
        sf = None
        sc = None
        tau_syn = 1/self.p.tau_syn_inv        
        for ts in range(self.seq_length):
            out_f, sf = self.features(x, sf)
            out_c, sc = self.classification(out_f, sc)
            voltages[ts, :, :] = out_c + 0.01 * torch.randn(x.shape[0], 10, device=x.device)

        y_hat, _ = torch.max(voltages, 0)
        return y_hat
    

    # Forward pass of a single batch
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc1, acc5 = self.__accuracy(y_hat, y, topk=(1, 5))
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc1', acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log('train_acc5', acc5, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc1, acc5 = self.__accuracy(y_hat, y, topk=(1, 5))
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc1', acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log('val_acc5', acc5, on_step=True, on_epoch=True)
        
    # The testing step is the same as the training, but with test data    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc1, acc5 = self.__accuracy(y_hat, y, topk=(1, 5))
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc1', acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log('test_acc5', acc5, on_step=True, on_epoch=True)

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
            optimizer, step_size=5, gamma=0.2
        )
        return optimizer

    @staticmethod
    def __accuracy(output, target, topk=(1, )):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


    
def main(args):

    # Set seeds
    torch.manual_seed(args.manual_seed)

    # Setup encoding
    num_channels = 3
    p = norse.torch.LIFParameters(v_th=torch.as_tensor(args.current_encoder_v_th))
    constant_encoder = norse.torch.ConstantCurrentLIFEncoder(
        seq_length=args.seq_length, p=p
    )
    if args.encoding == "poisson":
        encoder = norse.torch.PoissonEncoder(seq_length=args.seq_length, f_max=200)
    elif args.encoding == "constant":
        encoder = constant_encoder
    elif args.encoding == "constant_first":
        encoder = norse.torch.SpikeLatencyLIFEncoder(seq_length=args.seq_length, p=p)
    elif args.encoding == "signed_poisson":
        encoder = norse.torch.SignedPoissonEncoder(
            seq_length=args.seq_length, f_max=200
        )
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
        ]
    )
    transform_test = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root=".", train=True, download=True, transform=transform_train
        ),
        batch_size=args.batch_size,
        num_workers=32,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root=".", train=False, transform=transform_test),
        batch_size=args.batch_size,
        num_workers=32,        
    )

    # Define and train the model
    model = LIFConvNet(
        seq_length=args.seq_length,
        num_channels=num_channels,
        lr=args.lr,
        optimizer=args.optimizer,
        p=norse.torch.LIFParameters(v_th=torch.as_tensor(0.4)),
    )
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_dataloaders=test_loader)


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
        default=0.2,
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
        "--seq_length", default=128, type=int, help="Number of timesteps to do."
    )
    parser.add_argument(
        "--manual_seed", default=0, type=int, help="Random seed for torch"
    )
    args = parser.parse_args()

    main(args)
