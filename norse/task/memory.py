from absl import app, flags
import datetime
import logging
import numpy as np
import torch

from norse.torch.functional.encode import poisson_encode
from norse.torch.module.leaky_integrator import LICell
from norse.torch.module.lsnn import LSNNCell, LSNNParameters
from norse.torch.module.lif import LIFCell, LIFParameters

flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to use by pytorch.")
flags.DEFINE_integer("device_number", 0, "Index of the CUDA device to use, if at all.")
flags.DEFINE_float("dt", 0.001, "Time change per simulation step.")
flags.DEFINE_integer("epochs", 20, "Number of epochs.")
flags.DEFINE_float("learning_rate", 2e-3, "Learning rate to use.")
flags.DEFINE_integer(
    "log_interval", 10, "In which intervals to display learning progress."
)
flags.DEFINE_enum(
    "model",
    "super",
    ["super", "tanh", "circ", "logistic", "circ_dist"],
    "Model to use for training.",
)
flags.DEFINE_enum(
    "neuron_model",
    "lsnn",
    ["lsnn", "lif"],
    "Neuron model to use in network.",
)
flags.DEFINE_enum(
    "optimizer", "adam", ["adam", "sgd"], "Optimizer to use for training."
)
flags.DEFINE_boolean("plot", False, "Do intermediate plots during test sets.")
flags.DEFINE_enum(
    "plot_output",
    "show",
    ["show", "file"],
    "How to output plots: display on screen (show) or save to file (file).",
)

flags.DEFINE_integer(
    "poisson_rate",
    50,
    "Number of spikes per second, drawn from a poisson distribution.",
)
flags.DEFINE_integer(
    "population_size", 5, "Number of neurons per input bit (population encoded)."
)
flags.DEFINE_integer(
    "random_seed", int(torch.randint(high=100000, size=(1,))[0]), "Random seed."
)
flags.DEFINE_integer("samples", 1000, "Number of samples to use.")
flags.DEFINE_boolean("save_model", True, "Save the model after training every epoch.")
flags.DEFINE_integer("seq_length", 200, "Number of time steps per experiment step.")
flags.DEFINE_integer(
    "seq_steps", 24, "Number of steps in each experiment (should be at least 8)."
)
flags.DEFINE_integer(
    "seq_repetitions",
    24,
    "Number of times a sequence is repeated.",
)
flags.DEFINE_float("weight_decay", 1e-5, "Weight decay (L2 regularisation penalty).")


class MemoryNet(torch.nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        seq_length,
        is_lsnn,
        dt=0.01,
        model="super",
    ):
        super(MemoryNet, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.seq_length = seq_length
        self.is_lsnn = is_lsnn
        if is_lsnn:
            p = LSNNParameters(method=model)
            self.layer = LSNNCell(input_features, input_features, p, dt=dt)
        else:
            p = LIFParameters(method=model)
            self.layer = LIFCell(input_features, input_features, dt=dt)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.readout = LICell(input_features, output_features)

    def forward(self, x):
        batch_size = x.shape[0]

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
            _, sr = self.readout(spikes, sr)
            seq_readouts.append(sr.v)
            if (index + 1) % self.seq_length == 0:
                step_spikes.append(torch.stack(seq_spikes))
                seq_spikes = []
                step_readouts.append(torch.stack(seq_readouts))
                seq_readouts = []
        spikes = torch.cat(step_spikes)
        readouts = torch.stack(step_readouts)
        return readouts, spikes


class MemoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples,
        steps,
        seq_length,
        seq_repetitions,
        population_size,
        device,
        poisson_rate=1,
        dt=0.001,
        generator=torch.default_generator,
    ):
        self.samples = samples
        self.steps = steps
        self.seq_length = seq_length
        self.seq_repetitions = seq_repetitions
        self.population_size = population_size
        self.poisson_rate = poisson_rate
        self.device = device
        self.dt = dt
        self.generator = generator

        self.store_indices = torch.randint(
            low=0,
            high=steps // 2,
            size=(samples, seq_repetitions),
            generator=generator,
        )
        self.recall_indices = torch.randint(
            low=steps // 2,
            high=steps,
            size=(samples, seq_repetitions),
            generator=generator,
        )

    def __len__(self):
        return self.samples

    def _generate_sequence(self, idx, rep_idx):
        data_pattern = torch.stack(
            [torch.randperm(2, generator=self.generator) for _ in range(self.steps)]
        ).byte()
        store_index = self.store_indices[idx][rep_idx]
        recall_index = self.recall_indices[idx][rep_idx]
        store_pattern = torch.zeros((self.steps, 1)).byte()
        recall_pattern = store_pattern.clone()
        label_pattern = torch.zeros((self.steps, 2)).byte()

        store_pattern[store_index] = 1
        recall_pattern[recall_index] = 1
        label_class = data_pattern[store_index].byte()
        label_pattern[store_index] = label_class
        label_pattern[recall_index] = label_class
        data_pattern[recall_index] = torch.zeros(2)

        input_pattern = torch.cat((data_pattern, store_pattern, recall_pattern), dim=1)
        input_pattern = input_pattern.repeat_interleave(self.population_size, dim=1)
        encoded = poisson_encode(
            input_pattern,
            seq_length=self.seq_length,
            f_max=self.poisson_rate,
            dt=self.dt,
        )
        encoded = torch.cat(encoded.chunk(self.steps, dim=1)).squeeze()
        return encoded.to(self.device), label_pattern.to(self.device)

    def __getitem__(self, idx):
        repetitions = [
            self._generate_sequence(idx, i) for i in range(self.seq_repetitions)
        ]
        return (
            torch.cat([rep[0] for rep in repetitions]),
            torch.cat([rep[1] for rep in repetitions]),
        )


def _memory_accuracy_loss(xs, ys):
    xs = torch.stack(xs.mean(1).softmax(2).unbind(dim=1), dim=0)

    loss = torch.nn.functional.binary_cross_entropy(xs, ys.float())

    target = ys.argmax(1).min(1).indices
    batch_indices = ys.argmax(1).min(1).values
    recall_values = torch.stack(
        [xs[idx][batch_index] for idx, batch_index in enumerate(batch_indices)]
    ).argmax(1)
    return (target == recall_values).float().sum(), loss


def _plot_run(
    xs, ys, readouts, spikes, epoch, total_epochs, accuracy, file_prefix=None
):
    """
    Only plots the first batch event
    """

    def spikes_to_events(spike_list):
        return [[x for x, spike in enumerate(ts) if spike > 0] for ts in spike_list.T]

    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Setup grid
        plt.title(f"Accuracy: {accuracy} (epoch {epoch}/{total_epochs}")
        gridspec.GridSpec(4, 1)
        plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        input_events = spikes_to_events(xs[0])
        network_events = spikes_to_events(spikes[:, 0])
        plt.eventplot(
            network_events + input_events[::-1],
            linewidth=1,
            linelengths=0.8,
            color="black",
        )
        yticks = [x for x in range(len(network_events))] + [22.5, 27.5, 32.5, 37.5]
        y_labels = [""] * len(network_events)
        y_labels.append("Recall")
        y_labels.append("Store")
        y_labels.append("1")
        y_labels.append("0")
        plt.gca().set_yticks(yticks)
        plt.gca().set_yticklabels(y_labels)

        readout_events = readouts[:, :, 0].detach().reshape(-1, 2).softmax(1)
        readout_sum = (readout_events[:, 0] - readout_events[:, 1]).cpu().numpy()

        plt.subplot2grid((4, 1), (3, 0), rowspan=1)
        plt.gca().set_ylim(-1.1, 1.1)
        plt.gca().set_yticks([-1, 0, 1])
        plt.plot(readout_sum, color="black")

        plt.tight_layout()
        if file_prefix:
            plt.savefig(f"{file_prefix}_{epoch}.png")
        else:
            plt.show()
        plt.close()
    except ImportError as e:
        logging.warning("Plotting failed: Cannot import matplotlib: " + str(e))


def train(
    model,
    data_loader,
    optimizer,
    epoch,
    total_epochs,
    log_interval=1e10,
    writer=None,
):
    model.train()
    losses = []
    optimizer.zero_grad()
    step = 0

    for batch_idx, (xs, ys) in enumerate(data_loader):
        optimizer.zero_grad()
        readouts, _ = model(xs)
        accuracy, loss = _memory_accuracy_loss(readouts, ys)
        loss.backward()

        optimizer.step()
        step += 1

        logging.info(
            "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                total_epochs,
                batch_idx * len(xs),
                len(data_loader.dataset),
                100.0 * batch_idx / len(data_loader),
                loss.item(),
            )
        )

        if step % log_interval == 0 and writer:
            try:
                writer.add_scalar("Loss/train", loss.item(), step)
                writer.add_scalar("Accuracy/train", accuracy.item(), step)

                for tag, value in model.named_parameters():
                    tag = tag.replace(".", "/")
                    writer.add_histogram(tag, value.data.cpu().numpy(), step)
                    writer.add_histogram(
                        tag + "/grad", value.grad.data.cpu().numpy(), step
                    )
            except ValueError as e:
                logging.warning("Error when writing to tensorboard: " + str(e))

        losses.append(loss.item())

    mean_loss = torch.mean(torch.tensor(losses).float())
    return losses, mean_loss


def test(
    model,
    method,
    test_loader,
    epoch,
    total_epochs,
    plot=False,
    plot_file_prefix=None,
    writer=None,
):
    model.eval()
    test_loss = 0
    correct = 0
    did_plot = False
    runs = []

    with torch.no_grad():
        for xs, ys in test_loader:
            readouts, spikes = model(xs)
            if plot:
                runs.append((xs, ys, readouts, spikes))
            accuracy, loss = _memory_accuracy_loss(readouts, ys)
            test_loss += loss.item()
            correct += accuracy

            if plot and not did_plot:
                _plot_run(
                    xs,
                    ys,
                    readouts,
                    spikes,
                    epoch,
                    total_epochs,
                    accuracy,
                    plot_file_prefix,
                )
                did_plot = True

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)
    logging.info(
        f"\nTest set {method}: Average loss: {test_loss:.4f}, \
            Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n"
    )
    if writer:
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", accuracy, epoch)

    return test_loss, accuracy


def save(path, epoch, model, optimizer, is_best=False):
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "is_best": is_best,
        },
        path,
    )


def main(args):
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter()
    except ImportError:
        logging.info(
            "Disabling logging due to missing tensorboard dependency. Install tensorboard to enable logging."
        )
        writer = None

    FLAGS = flags.FLAGS

    input_features = (
        4 * FLAGS.population_size
    )  # (two bits + store + recall) * population size
    output_features = 2
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs

    generator = torch.random.manual_seed(FLAGS.random_seed)

    if FLAGS.device == "cuda":
        # Workaround for https://github.com/pytorch/pytorch/issues/21819
        torch.cuda.set_device(FLAGS.device_number)

    model = MemoryNet(
        input_features,
        output_features,
        seq_length=FLAGS.seq_length,
        is_lsnn=FLAGS.neuron_model == "lsnn",
        dt=FLAGS.dt,
    ).to(FLAGS.device)

    if FLAGS.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=FLAGS.learning_rate,
            momentum=0.9,
            weight_decay=FLAGS.weight_decay,
        )
    elif FLAGS.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay
        )

    file_prefix = f"memory_{datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%Hh%M')}_{FLAGS.neuron_model}"

    dataset = MemoryDataset(
        FLAGS.samples,
        FLAGS.seq_steps,
        FLAGS.seq_length,
        FLAGS.seq_repetitions,
        FLAGS.population_size,
        device=FLAGS.device,
        poisson_rate=FLAGS.poisson_rate,
        dt=FLAGS.dt,
        generator=generator,
    )
    dataset_test = MemoryDataset(
        FLAGS.samples // 5,
        FLAGS.seq_steps,
        FLAGS.seq_length,
        FLAGS.seq_repetitions,
        FLAGS.population_size,
        device=FLAGS.device,
        poisson_rate=FLAGS.poisson_rate,
        dt=FLAGS.dt,
        generator=generator,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

    training_losses = []
    mean_losses = []
    test_losses = []
    accuracies = []
    for epoch_index in range(epochs):
        training_loss, mean_loss = train(
            model,
            loader,
            optimizer,
            epoch_index,
            epochs,
            log_interval=FLAGS.log_interval,
            writer=writer,
        )
        test_loss, accuracy = test(
            model,
            FLAGS.model,
            loader_test,
            epoch_index,
            epochs,
            plot=FLAGS.plot,
            plot_file_prefix=(file_prefix if FLAGS.plot_output == "file" else None),
            writer=writer,
        )

        training_losses += training_loss
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        max_accuracy = np.max(np.array(accuracies))

        if FLAGS.save_model:
            model_path = f"{file_prefix}_epoch_{epoch_index}.pt"
            save(
                model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch_index,
                is_best=accuracy > max_accuracy,
            )

    np.save("training_losses.npy", np.array(training_losses))
    np.save("mean_losses.npy", np.array(mean_losses))
    np.save("test_losses.npy", np.array(test_losses))
    np.save("accuracies.npy", np.array(accuracies))
    model_path = f"{file_prefix}_final.pt"
    save(
        model_path,
        epoch=epoch_index,
        model=model,
        optimizer=optimizer,
        is_best=accuracy > max_accuracy,
    )
    if writer:
        writer.close()


if __name__ == "__main__":
    app.run(main)
