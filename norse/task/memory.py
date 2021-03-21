from argparse import ArgumentParser
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 4697ac2... Corrected memory task
from typing import Any, Optional, Tuple
import math

import matplotlib
import matplotlib.colors as colors
=======

>>>>>>> 267ca63... Added memory dataset and reworked memory task
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.utils.data

from norse.dataset.memory import MemoryStoreRecallDataset
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lsnn import LSNNRecurrentCell, LSNNCell, LSNNParameters
from norse.torch.functional.leaky_integrator import LIParameters
<<<<<<< HEAD
from norse.torch.module.lif import LIFRecurrentCell, LIFParameters
<<<<<<< HEAD
=======
from norse.torch.module.lif import LIFCell, LIFRecurrentCell, LIFParameters
>>>>>>> 4697ac2... Corrected memory task
from norse.torch.utils.plot import plot_spikes_2d


def zero_diagonal_(tensor):
    with torch.no_grad():
        return tensor.fill_diagonal_(0.0)


def sparsify_(tensor, sparsity):
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))

    with torch.no_grad():
        for col_idx in range(cols):
            row_indices = torch.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor


class LSNNLIFNet(torch.nn.Module):
    def __init__(self, input_features, p_lsnn, p_lif, dt):
        super().__init__()
        assert input_features % 2 == 0, "Input features must be a whole number"
        self.neurons_per_layer = input_features // 2

<<<<<<< HEAD
<<<<<<< HEAD
        self.lsnn_cell = LSNNRecurrentCell(
            input_features, self.neurons_per_layer, p_lsnn, dt=dt
        )
        self.lif_cell = LIFRecurrentCell(
            input_features, self.neurons_per_layer, p_lif, dt=dt
        )

    def forward(self, input_spikes: torch.Tensor, state: Optional[Tuple[Any, Any]]):
        if state is None:
            lif_state = None
            lsnn_state = None
        else:
            lif_state, lsnn_state = state

        lif_out, lif_state = self.lif_cell(input_spikes, lif_state)
        lsnn_out, lsnn_state = self.lsnn_cell(input_spikes, lsnn_state)
        out_spikes = torch.cat((lif_out, lsnn_out), -1)
        return out_spikes, (lif_state, lsnn_state)
=======
from norse.torch.util.plot import plot_spikes_2d
>>>>>>> 267ca63... Added memory dataset and reworked memory task
=======
        self.linear_input = torch.nn.Linear(input_features, input_features)
=======
        self.linear_input = torch.nn.Linear(input_features, input_features, bias=False)
>>>>>>> 71c8e4b... Work on memory task
        torch.nn.init.normal_(self.linear_input.weight, mean=0, std=1 / input_features)
        self.recurrent_weights = torch.nn.Parameter()
        self.linear_recurrent = torch.nn.Linear(
            input_features, input_features, bias=False
        )
        torch.nn.init.normal_(
            self.linear_recurrent.weight, mean=0, std=1 / input_features
        )
        zero_diagonal_(self.linear_recurrent.weight)
        # sparsify_(self.linear_recurrent.weight, 0.8)
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
>>>>>>> 4697ac2... Corrected memory task


class MemoryNet(pl.LightningModule):
    def __init__(self, input_features, output_features, args):
        super(MemoryNet, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.seq_length = args.seq_length
<<<<<<< HEAD
<<<<<<< HEAD
        self.optimizer = args.optimizer
        self.learning_rate = args.learning_rate
        self.regularization_factor = args.regularization_factor
        self.regularization_target = args.regularization_target / (
            self.seq_length * args.seq_repetitions
        )
        self.log("Neuron model", args.neuron_model)
        p_lsnn = LSNNParameters(
            method=args.model,
            v_th=torch.as_tensor(0.5),
            tau_syn_inv=torch.as_tensor(1 / 6e-3),
            tau_mem_inv=torch.as_tensor(1 / 2e-2),
            tau_adapt_inv=torch.as_tensor(1 / 1200.0),
            beta=torch.as_tensor(1.8),
        )
        p_lif = LIFParameters(
            method=args.model,
            v_th=torch.as_tensor(0.5),
            tau_syn_inv=torch.as_tensor(1 / 6e-3),
            tau_mem_inv=torch.as_tensor(1 / 2e-2),
        )
        p_li = LIParameters(
            tau_syn_inv=torch.as_tensor(1 / 6e-3),
            tau_mem_inv=torch.as_tensor(1 / 2e-2),
        )
        if args.neuron_model == "lsnn":
            self.capture_b = False
            self.layer = LSNNRecurrentCell(input_features, input_features, p=p_lsnn)
        elif args.neuron_model == "lsnnlif":
            self.layer = LSNNLIFNet(
                input_features, p_lsnn=p_lsnn, p_lif=p_lif, dt=args.dt
            )
            self.capture_b = True
        else:
            self.layer = LIFRecurrentCell(
                input_features, input_features, p=p_lif, dt=args.dt
            )
            self.capture_b = False
        self.readout = LILinearCell(input_features, output_features, p=p_li)
        self.scheduler = None
=======
        self.is_lsnn = args.neuron_model == "lsnn"
=======
>>>>>>> 4697ac2... Corrected memory task
        self.optimizer = args.optimizer
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        p_lsnn = LSNNParameters(
            method=args.model,
            v_th=torch.as_tensor(0.3),
            tau_syn_inv=torch.as_tensor(1 / 7e-3),
            tau_mem_inv=torch.as_tensor(1 / 4e-2),
            tau_adapt_inv=torch.as_tensor(1 / 1.2),
        )
        p_lif = LIFParameters(
            method=args.model,
            v_th=torch.as_tensor(1.0),
            tau_syn_inv=torch.as_tensor(1 / 5e-3),
            tau_mem_inv=torch.as_tensor(1 / 2e-2),
        )
        p_li = LIParameters(
            tau_syn_inv=torch.as_tensor(1 / 5e-3),
            tau_mem_inv=torch.as_tensor(1 / 2e-2),
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
<<<<<<< HEAD
        self.dropout = torch.nn.Dropout(p=0.1)
        li_p = LIParameters(
            tau_syn_inv=torch.as_tensor(1 / 5e-3),
            tau_mem_inv=torch.as_tensor(1 / 1e-2),
        )
        self.readout = LILinearCell(input_features, output_features, p=li_p)
>>>>>>> 267ca63... Added memory dataset and reworked memory task
=======
        self.readout = LILinearCell(input_features, output_features, p=p_li)
>>>>>>> 4697ac2... Corrected memory task

    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
<<<<<<< HEAD
=======
                weight_decay=self.weight_decay,
>>>>>>> 267ca63... Added memory dataset and reworked memory task
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
<<<<<<< HEAD
            )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.3
        )
        return [optimizer], [self.scheduler]

=======
                weight_decay=self.weight_decay,
            )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.99
        )
        return [optimizer], [self.scheduler]

>>>>>>> 267ca63... Added memory dataset and reworked memory task
    def forward(self, x):
        sl = None
        sr = None
        seq_spikes = []
        step_spikes = []
        seq_readouts = []
        step_readouts = []
        seq_betas = []
        for index, x_step in enumerate(x.unbind(1)):
            spikes, sl = self.layer(x_step, sl)
            seq_spikes.append(spikes)
<<<<<<< HEAD
<<<<<<< HEAD
=======
            spikes = self.dropout(spikes)
>>>>>>> 267ca63... Added memory dataset and reworked memory task
=======
>>>>>>> 4697ac2... Corrected memory task
            v, sr = self.readout(spikes, sr)
            seq_readouts.append(v)
            if (index + 1) % self.seq_length == 0:
                step_spikes.append(torch.stack(seq_spikes))
                seq_spikes = []
                step_readouts.append(torch.stack(seq_readouts))
                seq_readouts = []
            if self.capture_b:
                seq_betas.append(sl[1].b.clone().detach().cpu())
        spikes = torch.cat(step_spikes)
        readouts = torch.stack(step_readouts)
<<<<<<< HEAD
<<<<<<< HEAD
        betas = torch.stack(seq_betas) if len(seq_betas) > 0 else None
        return spikes, readouts, betas

    def testing_step(self, batch, batch_idx):
        xs, ys = batch
        spikes, readouts, betas = self(xs)
        # Loss: Difference between recall activity and recall pattern
        seq_readouts = readouts.mean(1).softmax(2).permute(1, 0, 2)
        mask = ys.sum(2).gt(0)
        labels = ys[mask].float()
        predictions = seq_readouts[mask]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, labels)
        # Accuracy: Sum of correct patterns out of total
        accuracy = (ys[mask].argmax(1) == seq_readouts[mask].argmax(1)).float().mean()
        values = {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "LR": self.scheduler.get_last_lr()[0],
        }

        # Plot random batch
        random_index = torch.randint(0, len(xs), (1,)).item()
        figure = _plot_run(
            xs[random_index],
            readouts[:, :, random_index],
            spikes[:, random_index],
            betas[:, random_index] if betas is not None else None,
        )
        self.logger.experiment.add_figure("Test readout", figure, self.current_epoch)
        self.log_dict(values, self.current_epoch)

        # Early stopping when loss <= 0.05
        if loss <= 0.05:
            self.trainer.should_stop = True

        return loss

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        spikes, readouts, _ = self(xs)
        # Loss: Difference between recall activity and recall pattern
        seq_readouts = readouts.mean(1).softmax(2).permute(1, 0, 2)
        mask = ys.sum(2).gt(0)
        labels = ys[mask].float()
        predictions = seq_readouts[mask]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, labels)
        # Regularization
        loss_reg = (
            (spikes.mean(0).mean(0) - self.regularization_target) ** 2
            * self.regularization_factor
        ).sum()
        self.log("loss_reg", loss_reg, self.current_epoch)
        return loss + loss_reg

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        spikes, readouts, betas = self(xs)
        # Loss: Difference between recall activity and recall pattern
        seq_readouts = readouts.mean(1).softmax(2).permute(1, 0, 2)
        mask = ys.sum(2).gt(0)
        labels = ys[mask].float()
        predictions = seq_readouts[mask]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, labels)
        # Accuracy: Sum of correct patterns out of total
        accuracy = (ys[mask].argmax(1) == seq_readouts[mask].argmax(1)).float().mean()
        values = {
            "val_loss": loss,
            "val_accuracy": accuracy,
            "LR": self.scheduler.get_last_lr()[0],
        }

        # Plot random batch
        random_index = torch.randint(0, len(xs), (1,)).item()
        figure = _plot_run(
            xs[random_index],
            readouts[:, :, random_index],
            spikes[:, random_index],
            betas[:, random_index] if betas is not None else None,
        )
        self.logger.experiment.add_figure("Readout", figure, self.current_epoch)
        self.log_dict(values, self.current_epoch)

        # Early stopping when loss <= 0.05
        if loss <= 0.05:
            self.trainer.should_stop = True

        return loss


def _plot_run(xs, readouts, spikes, betas=None):
    """
    Only plots the first batch event
    """
    figure = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(5, 2, width_ratios=[100, 1])
    plt.subplots_adjust(wspace=0.03)
    ax = figure.add_subplot(gs[:2, :1])
    plot_spikes_2d(xs.flip(1))  # Flip order so commands are shown on top
    yticks = torch.tensor([2, 7, 12, 17])
    y_labels = []
    y_labels.append("Recall")
    y_labels.append("Store")
    y_labels.append("1")
    y_labels.append("0")
    ax.set_xlim(0, len(xs))
    ax.set_ylabel("Command")
    ax.set_yticks(yticks)
    ax.set_yticklabels(y_labels)

    if betas is not None:
        ax = figure.add_subplot(gs[2:4, :1])
        blues_map = matplotlib.cm.get_cmap("Blues")
        beta_cmap = colors.ListedColormap(blues_map(torch.linspace(0.0, 0.75, 256)))
        bhm = plt.imshow(
            betas.T,
            cmap=beta_cmap,
            aspect="auto",
            interpolation="none",
            vmin=0,
            vmax=10,
        )
        alphas = spikes.clone().cpu().detach()
        alphas[spikes < 1] = 0
        plot_spikes_2d(spikes, alpha=alphas.T, cmap="gray_r")
        hax = figure.add_subplot(gs[2:4, 1:2])
        plt.colorbar(bhm, cax=hax, pad=0.05, aspect=20, label=r"$\beta$ value")
    else:
        ax = figure.add_subplot(gs[2:4, :1])
        plot_spikes_2d(spikes)
    ax.set_ylabel("Activity")
    ax.set_yticks([0, 19])
    ax.set_yticklabels([1, 20])
    ax.set_xlim(0, len(xs))

    ax = figure.add_subplot(gs[4, :1])
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
    plt.plot(v1 - v2, color="black", label="Readout")
    plt.legend(loc="upper right")
    plt.tight_layout()
    return figure
=======
        return readouts, spikes
=======
        return spikes, readouts
>>>>>>> 4697ac2... Corrected memory task

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
        # pos_reg_loss = torch.nn.functional.relu(spikes.mean(0) - 0.3).sum() * 1e-4
        # neg_reg_loss = torch.nn.functional.relu(0.005 - spikes.mean(0)).sum() * 1e-2
        return loss  # + pos_reg_loss + neg_reg_loss

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
<<<<<<< HEAD
    ax.set_yticks([-1, 0, 1])
    values, indices = readouts.detach().cpu().view(-1, 2).softmax(1).max(1)
    plt.plot(values * (indices * 2 - 1), color="black")
    if display == "show":
        plt.show()
    elif display is not None:
        plt.savefig(f"{display}.png")

    plt.close()
>>>>>>> 267ca63... Added memory dataset and reworked memory task
=======
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
>>>>>>> 4697ac2... Corrected memory task


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
<<<<<<< HEAD
<<<<<<< HEAD
    dataset_val = MemoryStoreRecallDataset(
        args.samples // 2,
=======
    dataset_test = MemoryStoreRecallDataset(
=======
    dataset_val = MemoryStoreRecallDataset(
>>>>>>> 4697ac2... Corrected memory task
        args.samples // 5,
>>>>>>> 267ca63... Added memory dataset and reworked memory task
        args.seq_length,
        args.seq_periods,
        args.seq_repetitions,
        args.population_size,
        poisson_rate=args.poisson_rate,
        dt=args.dt,
    )
<<<<<<< HEAD
    dataset_test = MemoryStoreRecallDataset(
        args.samples // 2,
        args.seq_length,
        args.seq_periods,
        args.seq_repetitions * 2,  # Double repetitions
        args.population_size,
        poisson_rate=args.poisson_rate,
        dt=args.dt,
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        save_top_k=args.save_top_k, monitor="val_loss"
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint])
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, test_dataloaders=test_loader)


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
        "--samples", type=int, default=512, help="Number of data points to train on."
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=1,
        help="Number of top models to checkpoint. -1 for all.",
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
        default=200,
        help="Poisson rate encoding for the data generation in Hz.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=0, help="Random seed for PyTorch"
    )
    parser.add_argument(
        "--regularization_factor",
        type=int,
        default=1e-2,
        help="Scale for regularization loss.",
    )
    parser.add_argument(
        "--regularization_target",
        type=int,
        default=10,
        help="Target measure for regularization",
    )

=======
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

>>>>>>> 267ca63... Added memory dataset and reworked memory task
    args = parser.parse_args()
    main(args)
