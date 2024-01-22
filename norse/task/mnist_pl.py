r"""
In this task, we train a spiking convolutional network to learn the
MNIST digit recognition task.

This version uses the `PyTorch Lightning <https://pytorchlightning.ai/>`_ library
to reduce the amount of boilerplate code around logging, checkpointing, training, etc.
"""

from argparse import ArgumentParser

import torch
import torch.utils.data
import torchvision

# pytype: disable=import-error
import pytorch_lightning as pl

# pytype: enable=import-error

import norse.torch as norse


class LIFConvNet(pl.LightningModule):
    def __init__(
        self,
        input_features,
        seq_length,
        learning_rate,
        model="super",
        optimizer="adam",
        input_scale=1.0,
        only_output=False,
        only_first_spike=False,
    ):
        super(LIFConvNet, self).__init__()
        if only_first_spike:
            self.encoder = norse.SpikeLatencyLIFEncoder(seq_length=seq_length)
        else:
            self.encoder = norse.ConstantCurrentLIFEncoder(seq_length=seq_length)
        self.input_features = input_features
        self.input_scale = input_scale
        self.learning_rate = learning_rate
        self.only_output = only_output
        self.optimizer = optimizer
        self.rsnn = norse.ConvNet4(method=model)
        self.seq_length = seq_length

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x.view(-1, self.input_features))
        x = x.reshape(self.seq_length, batch_size, 1, 28, 28)
        voltages = self.rsnn(x)
        m, _ = torch.max(voltages, 0)
        return torch.nn.functional.log_softmax(m, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = torch.nn.functional.nll_loss(out, y)
        return loss

    def configure_optimizers(self):
        if self.optimizer == "adam":
            opt = torch.optim.Adam
        else:
            opt = torch.optim.SGD

        if self.only_output:
            return opt(
                self.rsnn.out.parameters(), lr=self.learning_rate, weight_decay=1e-6
            )

        return opt(self.rsnn.parameters(), lr=self.learning_rate, weight_decay=1e-6)


def main(args):
    # First we create and transform the dataset
    data_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=".",
            train=True,
            download=True,
            transform=data_transform,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=".",
            train=False,
            transform=data_transform,
        ),
        batch_size=args.batch_size,
    )

    # Second, we create the PyTorch Lightning module
    model = LIFConvNet(
        28 * 28,  # Standard MNIST size
        seq_length=args.seq_length,
        learning_rate=args.learning_rate,
        model=args.model,
        optimizer=args.optimizer,
        only_output=args.train_only_output,
        only_first_spike=args.only_first_spike,
    )

    # Finally, we define the trainer and fit the model
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    parser = ArgumentParser("MNIST digit classification with spiking neural networks")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=1000, auto_select_gpus=True, progress_bar_refresh_rate=1
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Number of examples in one minibatch"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-3, help="Learning rate to use."
    )
    parser.add_argument(
        "--model",
        default="super",
        choices=["super", "tanh", "circ", "logistic", "circ_dist"],
        help="Model to use for training.",
    )
    parser.add_argument(
        "--only_first_spike",
        type=bool,
        default=False,
        help="Only one spike per input (latency coding).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--train_only_output",
        type=bool,
        default=False,
        help="Train only the last layer",
    )
    parser.add_argument(
        "--seq_length", default=200, type=int, help="Number of timesteps to do."
    )
    args = parser.parse_args()

    main(args)
