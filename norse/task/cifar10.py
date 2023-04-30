from argparse import ArgumentParser
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# pytype: disable=import-error
import pytorch_lightning as pl

# pytype: enable=import-error

from norse.torch.module import LIFCell, SequentialState, LICell
from norse.torch.functional.lif import LIFParameters


def label_smoothing_loss(y_hat, y, alpha=0.2):
    log_probs = F.log_softmax(y_hat, dim=1, _stacklevel=5)
    xent = F.nll_loss(log_probs, y, reduction="none")
    KL = -log_probs.mean(dim=1)
    loss = (1 - alpha) * xent + alpha * KL
    return loss.sum()


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

        c = 64
        c = [c, 2 * c, 4 * c, 4 * c]

        self.features = SequentialState(
            # preparation
            nn.Conv2d(
                num_channels, c[0], kernel_size=3, stride=1, padding=1, bias=False
            ),
            LIFCell(p),
            # block 1
            nn.Conv2d(c[0], c[1], kernel_size=3, stride=1, padding=1, bias=False),
            LIFCell(p),
            nn.MaxPool2d(2),
            # block 2
            nn.Conv2d(c[1], c[2], kernel_size=3, stride=1, padding=1, bias=False),
            LIFCell(p),
            nn.MaxPool2d(2),
            # block 3
            nn.Conv2d(c[2], c[3], kernel_size=3, stride=1, padding=1, bias=False),
            LIFCell(p),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        self.classification = SequentialState(
            # Classification
            nn.Linear(4096, 10, bias=False),
            LICell(),
        )

    def forward(self, x):
        voltages = torch.empty(
            self.seq_length, x.shape[0], 10, device=x.device, dtype=x.dtype
        )
        sf = None
        sc = None
        for ts in range(self.seq_length):
            out_f, sf = self.features(x, sf)
            # print(out_f.shape)
            out_c, sc = self.classification(out_f, sc)
            voltages[ts, :, :] = out_c + 0.001 * torch.randn(
                x.shape[0], 10, device=x.device
            )

        y_hat, _ = torch.max(voltages, 0)
        return y_hat

    # Forward pass of a single batch
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = label_smoothing_loss(y_hat, y)
        acc1, acc5 = self.__accuracy(y_hat, y, topk=(1, 5))
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "train_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True
        )
        self.log("train_acc5", acc5, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc1, acc5 = self.__accuracy(y_hat, y, topk=(1, 5))
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log("val_acc5", acc5, on_step=True, on_epoch=True)

    # The testing step is the same as the training, but with test data
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc1, acc5 = self.__accuracy(y_hat, y, topk=(1, 5))
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.log("test_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log("test_acc5", acc5, on_step=True, on_epoch=True)

    def training_epoch_end(self, outputs):
        if self.lr_step:
            self.scheduler.step()

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=1e-5
            )
        else:
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=5e-4 * self.batch_size,
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.2
        )
        return optimizer

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
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
    # Setup encoding
    num_channels = 3

    # Load datasets
    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
        ]
    )
    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
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
        p=LIFParameters(v_th=torch.as_tensor(0.4)),
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
        default=False,
        help="Use a stepper to reduce learning rate.",
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
    args = parser.parse_args()

    main(args)
