import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import pytorch_lightning as pl
from numpy.f2py.cfuncs import callbacks
from pytorch_lightning.cli import LightningCLI
from norse.torch import LIFCell, SequentialState, LICell
from norse.torch.functional.lif import LIFParameters
from norse.torch.functional.parameter import DEFAULT_BIO_PARAMS
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np


# Loss function
def label_smoothing_loss(y_hat, y, alpha=0.2):
    log_probs = F.log_softmax(y_hat, dim=1)
    xent = F.nll_loss(log_probs, y, reduction="none")
    KL = -log_probs.mean(dim=1)
    return ((1 - alpha) * xent + alpha * KL).sum()


# Model class
class LIFConvNet(pl.LightningModule):
    def __init__(self, seq_length: int = 1,
                 lr: float = 0.002,
                 p: dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in
                            DEFAULT_BIO_PARAMS['lif'].items()},
                 noise_scale=1e-6,
                 lr_step=True, num_channels: int = 3, optimizer: str = "adam"):
        super().__init__()
        self.save_hyperparameters()

        # LIF-Parameter korrekt laden
        self.p = LIFParameters(**p)

        self.lr = lr
        self.lr_step = lr_step
        self.optimizer = optimizer
        self.seq_length = seq_length

        c = 64
        c = [c, 2 * c, 4 * c, 4 * c]

        self.features = SequentialState(
            nn.Conv2d(num_channels, c[0], kernel_size=3, stride=1, padding=1, bias=False),
            LIFCell(self.p),
            nn.Conv2d(c[0], c[1], kernel_size=3, stride=1, padding=1, bias=False),
            LIFCell(self.p),
            nn.MaxPool2d(2),
            nn.Conv2d(c[1], c[2], kernel_size=3, stride=1, padding=1, bias=False),
            LIFCell(self.p),
            nn.MaxPool2d(2),
            nn.Conv2d(c[2], c[3], kernel_size=3, stride=1, padding=1, bias=False),
            LIFCell(self.p),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        self.classification = SequentialState(
            nn.Linear(4096, 10, bias=False),
            LICell(),
        )

    def forward(self, x):
        voltages = torch.empty(self.hparams.seq_length, x.shape[0], 10, device=x.device)
        sf, sc = None, None
        for ts in range(self.hparams.seq_length):
            out_f, sf = self.features(x, sf)
            out_c, sc = self.classification(out_f, sc)
            voltages[ts] = out_c + 0.001 * torch.randn_like(out_c)
        return voltages.max(dim=0)[0]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = label_smoothing_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr) if self.hparams.optimizer == 'adam' \
            else torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        return optimizer


# Data Module
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
        ])
        self.transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    def prepare_data(self):
        torchvision.datasets.CIFAR10(root=".", train=True, download=True)
        torchvision.datasets.CIFAR10(root=".", train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = torchvision.datasets.CIFAR10(root=".", train=True, transform=self.transform_train)
        self.val_dataset = torchvision.datasets.CIFAR10(root=".", train=False, transform=self.transform_test)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                           persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4,
                                           persistent_workers=True)



def plot_predictions(model, dataloader, device):
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)


    with torch.no_grad():
        y_hat = model(images)
        _, preds = torch.max(y_hat, 1)


    fig = plt.figure(figsize=(10, 5))
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        ax.imshow(img)
        ax.set_title(
            f"True: {labels[i].item()}, Pred: {preds[i].item()}")
    plt.show()


if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="best_model",
        save_top_k=1,
        mode="min",
        save_weights_only=True,
        verbose=True
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Trainer aus LightningCLI
    # cli = LightningCLI(LIFConvNet, CIFAR10DataModule, trainer_defaults={ "callbacks" : [checkpoint_callback]} )

    #torch.save(cli.model,'trained_model.pth')

    model = LIFConvNet.load_from_checkpoint("checkpoints\\best_model.ckpt")
    model.to(device)


    cifar10_dm = CIFAR10DataModule(batch_size=32)
    cifar10_dm.setup()
    dataloader = cifar10_dm.val_dataloader()


    plot_predictions(model, dataloader, device)
