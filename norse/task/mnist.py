import os
import uuid

from absl import app
from absl import flags
from absl import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from norse.torch.models.conv import ConvNet4
from norse.torch.module.encode import ConstantCurrentLIFEncoder

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "only_first_spike", False, "Only one spike per input (latency coding)."
)
flags.DEFINE_bool("save_grads", False, "Save gradients of backward pass.")
flags.DEFINE_integer(
    "grad_save_interval", 10, "Interval for gradient saving of backward pass."
)
flags.DEFINE_bool("refrac", False, "Use refractory time.")
flags.DEFINE_integer("plot_interval", 10, "Interval for plotting.")
flags.DEFINE_float("input_scale", 1, "Scaling factor for input current.")
flags.DEFINE_bool(
    "find_learning_rate", False, "Use learning rate finder to find learning rate."
)
flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to use by pytorch.")
flags.DEFINE_integer("epochs", 10, "Number of training episodes to do.")
flags.DEFINE_integer("seq_length", 200, "Number of timesteps to do.")
flags.DEFINE_integer("batch_size", 32, "Number of examples in one minibatch.")
flags.DEFINE_enum(
    "model",
    "super",
    ["super", "tanh", "circ", "logistic", "circ_dist"],
    "Model to use for training.",
)
flags.DEFINE_string("prefix", "", "Prefix to use for saving the results")
flags.DEFINE_enum(
    "optimizer", "adam", ["adam", "sgd"], "Optimizer to use for training."
)
flags.DEFINE_bool("clip_grad", False, "Clip gradient during backpropagation")
flags.DEFINE_float("grad_clip_value", 1.0, "Gradient to clip at.")
flags.DEFINE_float("learning_rate", 2e-3, "Learning rate to use.")
flags.DEFINE_integer(
    "log_interval", 10, "In which intervals to display learning progress."
)
flags.DEFINE_integer("model_save_interval", 50, "Save model every so many epochs.")
flags.DEFINE_boolean("save_model", True, "Save the model after training.")
flags.DEFINE_boolean("big_net", False, "Use bigger net...")
flags.DEFINE_boolean("only_output", False, "Train only the last layer...")
flags.DEFINE_boolean("do_plot", False, "Do intermediate plots")
flags.DEFINE_integer("random_seed", 1234, "Random seed to use")


class LIFConvNet(torch.nn.Module):
    def __init__(
        self,
        input_features,
        seq_length,
        model="super",
        only_first_spike=False,
    ):
        super(LIFConvNet, self).__init__()
        self.constant_current_encoder = ConstantCurrentLIFEncoder(seq_length=seq_length)
        self.only_first_spike = only_first_spike
        self.input_features = input_features
        self.rsnn = ConvNet4(method=model)
        self.seq_length = seq_length

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.constant_current_encoder(
            x.view(-1, self.input_features) * FLAGS.input_scale
        )
        if self.only_first_spike:
            # delete all spikes except for first
            zeros = torch.zeros_like(x.cpu()).detach().numpy()
            idxs = x.cpu().nonzero().detach().numpy()
            spike_counter = np.zeros((FLAGS.batch_size, 28 * 28))
            for t, batch, nrn in idxs:
                if spike_counter[batch, nrn] == 0:
                    zeros[t, batch, nrn] = 1
                    spike_counter[batch, nrn] += 1
            x = torch.from_numpy(zeros).to(x.device)

        x = x.reshape(self.seq_length, batch_size, 1, 28, 28)
        voltages = self.rsnn(x)
        m, _ = torch.max(voltages, 0)
        log_p_y = torch.nn.functional.log_softmax(m, dim=1)
        return log_p_y


def train(model, device, train_loader, optimizer, epoch, writer=None):
    model.train()
    losses = []

    batch_len = len(train_loader)
    step = batch_len * epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()

        if FLAGS.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_clip_value)

        optimizer.step()
        step += 1

        if batch_idx % FLAGS.log_interval == 0:
            logging.info(
                "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    FLAGS.epochs,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

        if step % FLAGS.log_interval == 0 and writer:
            _, argmax = torch.max(output, 1)
            accuracy = (target == argmax.squeeze()).float().mean()

            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Accuracy/train", accuracy.item(), step)

            for tag, value in model.named_parameters():
                tag = tag.replace(".", "/")
                writer.add_histogram(tag, value.data.cpu().numpy(), step)
                writer.add_histogram(tag + "/grad", value.grad.data.cpu().numpy(), step)

        if FLAGS.do_plot and batch_idx % FLAGS.plot_interval == 0:
            ts = np.arange(0, FLAGS.seq_length)
            fig, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
            axs = axs.reshape(-1)  # flatten
            for nrn in range(10):
                one_trace = model.voltages.detach().cpu().numpy()[:, 0, nrn]
                fig.sca(axs[nrn])
                fig.plot(ts, one_trace)
            fig.xlabel("Time [s]")
            fig.ylabel("Membrane Potential")

            writer.add_figure("Voltages/output", fig, step)

        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss


def test(model, device, test_loader, epoch, writer=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)
    logging.info(
        f"\nTest set {FLAGS.model}: Average loss: {test_loss:.4f}, \
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


def load(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()
    return model, optimizer


def main(argv):
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter()
    except ImportError:
        writer = None

    torch.manual_seed(FLAGS.random_seed)

    np.random.seed(FLAGS.random_seed)
    if hasattr(torch, "cuda_is_available"):
        if torch.cuda_is_available():
            torch.cuda.manual_seed(FLAGS.random_seed)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    device = torch.device(FLAGS.device)

    kwargs = {"num_workers": 1, "pin_memory": True} if FLAGS.device == "cuda" else {}
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=".",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    # torchvision.transforms.
                    #    RandomCrop(size=[28,28], padding=4)
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=FLAGS.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=".",
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=FLAGS.batch_size,
        **kwargs,
    )

    label = os.environ.get("SLURM_JOB_ID", str(uuid.uuid4()))
    if FLAGS.prefix:
        path = f"runs/mnist/{FLAGS.prefix}/{label}"
    else:
        path = f"runs/mnist/{label}"

    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    FLAGS.append_flags_into_file("flags.txt")

    input_features = 28 * 28

    model = LIFConvNet(
        input_features,
        FLAGS.seq_length,
        model=FLAGS.model,
        only_first_spike=FLAGS.only_first_spike,
    ).to(device)

    if FLAGS.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
    elif FLAGS.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    if FLAGS.only_output:
        optimizer = torch.optim.Adam(model.out.parameters(), lr=FLAGS.learning_rate)

    training_losses = []
    mean_losses = []
    test_losses = []
    accuracies = []

    for epoch in range(FLAGS.epochs):
        training_loss, mean_loss = train(
            model, device, train_loader, optimizer, epoch, writer=writer
        )
        test_loss, accuracy = test(model, device, test_loader, epoch, writer=writer)

        training_losses += training_loss
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        max_accuracy = np.max(np.array(accuracies))

        if (epoch % FLAGS.model_save_interval == 0) and FLAGS.save_model:
            model_path = f"mnist-{epoch}.pt"
            save(
                model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                is_best=accuracy > max_accuracy,
            )

    np.save("training_losses.npy", np.array(training_losses))
    np.save("mean_losses.npy", np.array(mean_losses))
    np.save("test_losses.npy", np.array(test_losses))
    np.save("accuracies.npy", np.array(accuracies))
    model_path = "mnist-final.pt"
    save(
        model_path,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        is_best=accuracy > max_accuracy,
    )
    if writer:
        writer.close()


if __name__ == "__main__":
    app.run(main)
