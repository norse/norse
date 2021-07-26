"""
An example of using sparse event-based Dynamic Vision System (DVS) data with 
NVIDIA's MinkowskiEngine to implement sparse convolutions in spiking neural
networks.

See https://github.com/NVIDIA/MinkowskiEngine
"""

from argparse import ArgumentParser

import torch
import numpy as np
import MinkowskiEngine as ME
import pytorch_lightning as pl

# Tonic is a great library for event-based datasets
#   https://github.com/neuromorphs/tonic
import tonic
import tonic.transforms as tonic_transforms

import norse.torch as norse

from minkowski.lif import MinkowskiLIFCell


class NMNISTReLUNetwork(torch.nn.Module):
    def __init__(self, in_channels, out_channels=128, out_features=10, D=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            ME.MinkowskiConvolution(in_channels, 32, 5, dimension=D),
            ME.MinkowskiConvolution(32, 64, 5, stride=2, dimension=D),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(64, out_channels, kernel_size=3, dimension=D),
            ME.MinkowskiReLU(),
            ME.MinkowskiGlobalPooling(),
            ME.MinkowskiLinear(128, out_features),
            ME.MinkowskiReLU(),
            ME.MinkowskiSoftmax(),
            ME.MinkowskiToFeature(),
        )

    def forward(self, x):
        # Force BCTXY format
        dense = x.to_dense().permute(0, 2, 1, 3, 4)
        tensor = ME.to_sparse(dense)
        out = self.net(tensor)
        return out


class NMNISTModule(pl.LightningModule):
    def __init__(
        self,
        model,
        batch_size,
        transform,
        lr=1e-2,
        weight_decay=1e-5,
        data_root="./data",
    ):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.data_root = data_root

        self.criterion = torch.nn.CrossEntropyLoss()
        self.transform = transform

    def forward(self, x):
        return self.model(x)

    # def batch_collate_fn(self, list_data):
    #     batches = []
    #     for batch_idx in range(len(list_data[0][0])):
    #         chunks = []
    #         for datapoint in list_data:
    #             if batch_idx < len(datapoint[0]):
    #                 chunks.append(datapoint[0][batch_idx])
    #         len_diff = self.batch_size - len(chunks)
    #         while len_diff > 0:
    #             chunks.append(
    #                 [
    #                     torch.zeros((1, 20), dtype=torch.float32),
    #                     torch.zeros((1, 2), dtype=torch.float32),
    #                 ]
    #             )
    #             len_diff -= 1
    #         batches.append(ME.utils.batch_sparse_collate(chunks))
    #     labels = [x[1] for x in list_data]
    #     return batches, torch.stack(labels).squeeze()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            tonic.datasets.NMNIST(
                save_to=self.data_root,
                train=True,
                transform=self.transform,
            ),
            collate_fn=tonic.utils.pad_tensors,
            batch_size=self.batch_size,
            shuffle=True,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         tonic.datasets.NMNIST(save_to="./data", train=False, transform=transform),
    #         collate_fn=single_chunk_collate
    #         if self.timesteps <= 1
    #         else batch_collate_fn(self.batch_size),
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #     )

    def training_step(self, batch, batch_idx):
        chunks, labels = batch
        # Must clear cache at regular interval
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        out = self(chunks)
        loss = self.criterion(out, labels)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     chunks, labels = batch
    #     # Must clear cache at regular interval
    #     if self.global_step % 10 == 0:
    #         torch.cuda.empty_cache()
    #     out, _ = self(chunks)
    #     loss = self.criterion(out, labels)
    #     acc = torch.eq(out.detach().argmax(1), labels).float().mean()
    #     self.log("VAcc", acc)
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        return [optimizer], [scheduler]


def main(args):

    transform = tonic_transforms.Compose(
        [
            tonic_transforms.Denoise(filter_time=10000),
            tonic_transforms.Subsample(args.subsample),
            tonic_transforms.ToSparseTensor(),
        ]
    )

    network = NMNISTReLUNetwork(in_channels=2)
    module = NMNISTModule(network, 8, transform=transform, data_root=args.data_root)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(module)


if __name__ == "__main__":
    parser = ArgumentParser("N-MNIST training")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--subsample",
        type=float,
        default=1e-3,
        help="Supsampling multiplier for timesteps. 0.1 = 10\% of timesteps. Defaults to 1e-3",
    )
    parser.add_argument(
        "--model",
        default="snn",
        choices=["snn", "ann"],
        help="Model to use for training. Defaults to snn.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="The root of data for the NMNIST dataset. Defaults to ./data",
    )
    args = parser.parse_args()
    main(args)
