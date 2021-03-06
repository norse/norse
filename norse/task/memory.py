from argparse import ArgumentParser

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.utils.data

from norse.dataset.memory import MemoryStoreRecallDataset
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lsnn import LSNNRecurrentCell, LSNNParameters
from norse.torch.functional.leaky_integrator import LIParameters
from norse.torch.module.lif import LIFRecurrentCell, LIFParameters
from norse.torch.util.plot import plot_spikes_2d


class MemoryNet(pl.LightningModule):
    def __init__(self, input_features, output_features, args):
        super(MemoryNet, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.seq_length = args.seq_length
        self.is_lsnn = args.neuron_model == "lsnn"
        self.optimizer = args.optimizer
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.plot = args.plot
        if self.is_lsnn:
            p = LSNNParameters(
                method=args.model,
                beta=torch.nn.Parameter(torch.as_tensor(1.8)),
                tau_syn_inv=torch.nn.Parameter(torch.as_tensor(1.0 / 5e-3)),
                tau_mem_inv=torch.nn.Parameter(torch.as_tensor(1.0 / 1e-2)),
                tau_adapt_inv=torch.nn.Parameter(torch.as_tensor(1.0 / 700)),
            )
            self.layer = LSNNRecurrentCell(
                input_features, input_features, p=p, dt=args.dt
            )
        else:
            p = LIFParameters(
                method=args.model,
                tau_syn_inv=torch.nn.Parameter(torch.as_tensor(1.0 / 5e-3)),
                tau_mem_inv=torch.nn.Parameter(torch.as_tensor(1.0 / 1e-2)),
            )
            self.layer = LIFRecurrentCell(
                input_features, input_features, p=p, dt=args.dt
            )
        self.dropout = torch.nn.Dropout(p=0.1)
        li_p = LIParameters(
            tau_syn_inv=torch.as_tensor(1 / 5e-3),
            tau_mem_inv=torch.as_tensor(1 / 1e-2),
        )
        self.readout = LILinearCell(input_features, output_features, p=li_p)

    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        return optimizer

    def forward(self, x):
        sl = None
        sr = None
        seq_spikes = []
        step_spikes = []
        seq_readouts = []
        step_readouts = []
        for index, x_step in enumerate(x.unbind(1)):
            spikes, sl = self.layer(x_step, sl)
            seq_spikes.append(spikes)
            spikes = self.dropout(spikes)
            v, sr = self.readout(spikes, sr)
            seq_readouts.append(v)
            if (index + 1) % self.seq_length == 0:
                step_spikes.append(torch.stack(seq_spikes))
                seq_spikes = []
                step_readouts.append(torch.stack(seq_readouts))
                seq_readouts = []
        spikes = torch.cat(step_spikes)
        readouts = torch.stack(step_readouts)
        return readouts, spikes

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        readouts, _ = self(xs)
        prediction = readouts.max(1)[0].softmax(2).permute(1, 0, 2)
        loss = torch.nn.functional.binary_cross_entropy(prediction, ys.float())
        self.log("loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        readouts, spikes = self(xs)
        prediction = readouts.max(1)[0].softmax(2).permute(1, 0, 2)
        loss = torch.nn.functional.binary_cross_entropy(prediction, ys.float())
        mask = ys.sum(2).gt(0)
        accuracy = (ys[mask].argmax(1) == prediction[mask].argmax(1)).float().mean()
        values = {"loss": loss, "accuracy": accuracy}

        if self.plot is not None:
            _plot_run(
                xs[0],
                ys[0],
                readouts[0],
                spikes[:, 0],
                accuracy,
                display=self.current_epoch if self.plot == "file" else self.plot,
            )
            torch.save([readouts], f"{self.current_epoch}.pt")

        self.log_dict(values)
        return values


def _plot_run(xs, ys, readouts, spikes, accuracy, display=None):
    """
    Only plots the first batch event
    """
    plt.figure(figsize=(16, 10))
    plt.title(f"Accuracy: {accuracy}")
    gridspec.GridSpec(5, 1)
    ax = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
    plot_spikes_2d(xs)
    yticks = torch.tensor([2, 7, 12, 17])
    y_labels = []
    y_labels.append("0")
    y_labels.append("1")
    y_labels.append("Store")
    y_labels.append("Recall")
    ax.set_yticks(yticks)
    ax.set_yticklabels(y_labels)

    ax = plt.subplot2grid((5, 1), (2, 0), rowspan=2)
    plot_spikes_2d(spikes)
    ax.set_yticks([0, 19])
    ax.set_yticklabels([1, 20])

    ax = plt.subplot2grid((5, 1), (4, 0), rowspan=1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([-1, 0, 1])
    values, indices = readouts.detach().cpu().view(-1, 2).softmax(1).max(1)
    plt.plot(values * (indices * 2 - 1), color="black")
    if display == "show":
        plt.show()
    elif display is not None:
        plt.savefig(f"{display}.png")

    plt.close()


def main(args):
    input_features = (
        4 * args.population_size
    )  # (two bits + store + recall) * population size
    output_features = 2
    batch_size = args.batch_size
    torch.random.manual_seed(args.random_seed)

    model = MemoryNet(input_features, output_features, args)

    dataset = MemoryStoreRecallDataset(
        args.samples,
        args.seq_length,
        args.seq_periods,
        args.seq_repetitions,
        args.population_size,
        poisson_rate=args.poisson_rate,
        dt=args.dt,
    )
    dataset_test = MemoryStoreRecallDataset(
        args.samples // 5,
        args.seq_length,
        args.seq_periods,
        args.seq_repetitions,
        args.population_size,
        poisson_rate=args.poisson_rate,
        dt=args.dt,
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)
    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    parser = ArgumentParser("Memory task with spiking neural networks")
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
        "--neuron_model",
        type=str,
        default="lsnn",
        choices=["lsnn", "lif"],
        help="Neuron model to use in network.",
    )
    parser.add_argument(
        "--dt", default=0.001, type=str, help="Time change per simulation step."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        choices=[None, "show", "file"],
        help="How to output plots: display on screen (show) or save to file (file). Defaults to None (no plots)",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=5,
        help="Number of neurons per input bit (population encoded).",
    )
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of data points to train on."
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=200,
        help="Number of time steps in one command/value.",
    )
    parser.add_argument(
        "--seq_periods",
        type=int,
        default=4,
        help="Number of periods in one data point/iteration.",
    )
    parser.add_argument(
        "--seq_repetitions",
        type=int,
        default=4,
        help="Number of times one store/retrieve command is repeated in a single data point.",
    )
    parser.add_argument(
        "--poisson_rate",
        type=float,
        default=250,
        help="Poisson rate encoding for the data generation in Hz.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularisation penalty).",
    )
    parser.add_argument(
        "--random_seed", type=int, default=0, help="Random seed for PyTorch"
    )

    args = parser.parse_args()
    main(args)
