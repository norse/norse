from argparse import ArgumentParser
from typing import Any, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.utils.data

from norse.dataset.memory import MemoryStoreRecallDataset
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lsnn import LSNNRecurrentCell, LSNNCell, LSNNParameters
from norse.torch.functional.leaky_integrator import LIParameters
from norse.torch.module.lif import LIFCell, LIFRecurrentCell, LIFParameters
from norse.torch.utils.plot import plot_spikes_2d


class LSNNLIFNet(torch.nn.Module):
    def __init__(self, input_features, p_lsnn, p_lif, dt):
        super().__init__()
        assert input_features % 2 == 0, "Input features must be a whole number"
        self.neurons_per_layer = input_features // 2

        self.linear_input = torch.nn.Linear(input_features, input_features)
        torch.nn.init.normal_(self.linear_input.weight, mean=0, std=1 / input_features)
        self.linear_recurrent = torch.nn.Linear(input_features, input_features)
        torch.nn.init.normal_(
            self.linear_recurrent.weight, mean=0, std=1 / input_features
        )
        self.lsnn_cell = LSNNCell(p_lsnn, dt=dt)
        self.lif_cell = LIFCell(p_lif, dt=dt)

    def forward(
        self, input_spikes: torch.Tensor, state: Optional[Tuple[Any, Any, torch.Tensor]]
    ):
        if state is None:
            lif_state = None
            lsnn_state = None
            previous_spikes = torch.zeros_like(input_spikes)
        else:
            lif_state, lsnn_state, previous_spikes = state

        weighted_input = self.linear_input(input_spikes) + self.linear_recurrent(
            previous_spikes
        )
        lif_input, lsnn_input = weighted_input.split(self.neurons_per_layer, -1)
        lif_out, lif_state = self.lif_cell(lif_input, lif_state)
        lsnn_out, lsnn_state = self.lsnn_cell(lsnn_input, lsnn_state)
        out_spikes = torch.cat((lif_out, lsnn_out), -1)
        return out_spikes, (lif_state, lsnn_state, out_spikes)


class MemoryNet(pl.LightningModule):
    def __init__(self, input_features, output_features, args):
        super(MemoryNet, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.seq_length = args.seq_length
        self.optimizer = args.optimizer
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        p_lsnn = LSNNParameters(
            method=args.model,
            v_th=torch.as_tensor(0.5),
            tau_syn_inv=torch.nn.Parameter(torch.as_tensor(1 / 7e-3)),
            tau_mem_inv=torch.nn.Parameter(torch.as_tensor(1 / 4e-2)),
            tau_adapt_inv=torch.as_tensor(1 / 1.2),
        )
        p_lif = LIFParameters(
            method=args.model,
            v_th=torch.as_tensor(1.0),
            tau_syn_inv=torch.nn.Parameter(torch.as_tensor(1 / 5e-3)),
            tau_mem_inv=torch.nn.Parameter(torch.as_tensor(1 / 2e-2)),
        )
        p_li = LIParameters(
            tau_syn_inv=torch.nn.Parameter(torch.as_tensor(1 / 5e-3)),
            tau_mem_inv=torch.nn.Parameter(torch.as_tensor(1 / 2e-2)),
        )
        if args.neuron_model == "lsnn":
            self.layer = LSNNRecurrentCell(input_features, input_features, p=p_lsnn)
        elif args.neuron_model == "lsnnlif":
            self.layer = LSNNLIFNet(
                input_features, p_lsnn=p_lsnn, p_lif=p_lif, dt=args.dt
            )
        else:
            self.layer = LIFRecurrentCell(
                input_features, input_features, p=p_lif, dt=args.dt
            )
        self.readout = LILinearCell(input_features, output_features, p=p_li)

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

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.99
        )
        return [optimizer], [self.scheduler]

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
            v, sr = self.readout(spikes, sr)
            seq_readouts.append(v)
            if (index + 1) % self.seq_length == 0:
                step_spikes.append(torch.stack(seq_spikes))
                seq_spikes = []
                step_readouts.append(torch.stack(seq_readouts))
                seq_readouts = []
        spikes = torch.cat(step_spikes)
        readouts = torch.stack(step_readouts)
        return spikes, readouts

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        spikes, readouts = self(xs)
        # Loss: Difference between recall activity and recall pattern
        softmax = readouts.softmax(3).mean(1).permute(1, 0, 2)
        mask = ys.sum(2).gt(0)
        labels = ys[mask].float()
        predictions = softmax[mask]
        loss = torch.nn.functional.binary_cross_entropy(predictions, labels)
        # Regularization: Punish 0.05 < activity > 30%
        pos_reg_loss = torch.nn.functional.relu(spikes.mean(0) - 0.3).sum() * 1e-4
        neg_reg_loss = torch.nn.functional.relu(0.005 - spikes.mean(0)).sum() * 1e-2
        return loss + pos_reg_loss + neg_reg_loss

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        spikes, readouts = self(xs)
        # Loss: Difference between recall activity and recall pattern
        softmax = readouts.softmax(3).mean(1).permute(1, 0, 2)
        mask = ys.sum(2).gt(0)
        labels = ys[mask].float()
        predictions = softmax[mask]
        loss = torch.nn.functional.binary_cross_entropy(predictions, labels)
        # Accuracy: Sum of correct patterns out of total
        accuracy = (ys[mask].argmax(1) == softmax[mask].argmax(1)).float().mean()
        values = {
            "val_loss": loss,
            "val_accuracy": accuracy,
            "LR": self.scheduler.get_last_lr()[0],
        }

        # Plot random batch
        random_index = torch.randint(0, len(xs), (1,)).item()
        figure = _plot_run(
            xs[random_index],
            ys[random_index],
            readouts[:, :, random_index],
            spikes[:, random_index],
        )
        self.logger.experiment.add_figure("Readout", figure, self.current_epoch)
        self.log_dict(values, self.current_epoch)

        # Early stopping when accuracy >= 95%
        if accuracy >= 0.95:
            self.trainer.should_stop = True

        return loss


def _plot_run(xs, ys, readouts, spikes):
    """
    Only plots the first batch event
    """
    figure = plt.figure(figsize=(16, 10))
    gridspec.GridSpec(5, 1)
    ax = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
    plot_spikes_2d(xs.flip(1))  # Flip order so commands are shown on top
    yticks = torch.tensor([2, 7, 12, 17])
    y_labels = []
    y_labels.append("Recall")
    y_labels.append("Store")
    y_labels.append("1")
    y_labels.append("0")
    ax.set_ylabel("Command")
    ax.set_yticks(yticks)
    ax.set_yticklabels(y_labels)

    ax = plt.subplot2grid((5, 1), (2, 0), rowspan=2)
    plot_spikes_2d(spikes)
    ax.set_ylabel("Activity")
    ax.set_yticks([0, 19])
    ax.set_yticklabels([1, 20])

    ax = plt.subplot2grid((5, 1), (4, 0), rowspan=1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([-1, 1])
    ax.set_yticklabels([0, 1])
    v1, v2 = readouts.detach().cpu().view(-1, 2).softmax(1).chunk(2, 1)
    ax.set_ylabel("Readout")
    ax.set_xlim(0, len(v1))
    plt.plot(
        [0, len(xs)],
        [0, 0],
        color="black",
        linestyle="dashed",
        label="Decision boundary",
    )
    plt.plot(v1 - v2, color="black", label="Softmax readout")
    plt.legend(loc="upper right")
    return figure


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
    dataset_val = MemoryStoreRecallDataset(
        args.samples // 5,
        args.seq_length,
        args.seq_periods,
        args.seq_repetitions,
        args.population_size,
        poisson_rate=args.poisson_rate,
        dt=args.dt,
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = ArgumentParser("Memory task with spiking neural networks")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=1000, auto_select_gpus=True, progress_bar_refresh_rate=1
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Number of examples in one minibatch",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate to use."
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
        default="lsnnlif",
        choices=["lsnn", "lif", "lsnnlif"],
        help="Neuron model to use in network. 100%% LSNN, 100%% LIF and 50/50.",
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
        "--population_size",
        type=int,
        default=5,
        help="Number of neurons per input bit (population encoded).",
    )
    parser.add_argument(
        "--samples", type=int, default=250, help="Number of data points to train on."
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
        default=12,
        help="Number of commands in one data point/iteration.",
    )
    parser.add_argument(
        "--seq_repetitions",
        type=int,
        default=1,
        help="Number of times one store/retrieve command is repeated in a single data point.",
    )
    parser.add_argument(
        "--poisson_rate",
        type=float,
        default=100,
        help="Poisson rate encoding for the data generation in Hz.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-8,
        help="Weight decay (L2 regularisation penalty).",
    )
    parser.add_argument(
        "--random_seed", type=int, default=0, help="Random seed for PyTorch"
    )

    args = parser.parse_args()
    main(args)
