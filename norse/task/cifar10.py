import os
import datetime
import uuid

from absl import app
from absl import flags
from absl import logging

import torch

from collections import namedtuple
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from norse.torch.models.conv import ConvNet, ConvNet4
from norse.torch.module.if_current_encoder import IFConstantCurrentEncoder

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "only_first_spike", False, "Only one spike per input (latency coding)."
)
flags.DEFINE_bool("save_grads", False, "Save gradients of backward pass.")
flags.DEFINE_integer(
    "grad_save_interval", 10, "Interval for gradient saving of backward pass."
)
flags.DEFINE_string("prefix", "", "Prefix for save path to use.")
flags.DEFINE_enum(
    "encoding",
    "poisson",
    ["poisson", "constant", "constant_polar", "signed_poisson", "signed_constant"],
    "Encoding to use for input",
)

flags.DEFINE_enum(
    "net", "convnet4", ["convnet", "convnet4"], "Which network architecture to use"
)
flags.DEFINE_integer("plot_interval", 10, "Interval for plotting.")
flags.DEFINE_float("input_scale", 1, "Scaling factor for input current.")
flags.DEFINE_float("current_encoder_v_th", 1.0, "v_th for constant current encoder")
flags.DEFINE_bool("learning_rate_schedule", False, "Use a learning rate schedule")
flags.DEFINE_bool("find_learning_rate", False, "Find learning rate")

flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to use by pytorch.")
flags.DEFINE_integer("epochs", 10, "Number of training episodes to do.")
flags.DEFINE_integer("seq_length", 200, "Number of timesteps to do.")
flags.DEFINE_integer("batch_size", 32, "Number of examples in one minibatch.")
flags.DEFINE_integer("hidden_size", 100, "Number of neurons in the hidden layer.")
flags.DEFINE_enum(
    "model",
    "super",
    ["super", "tanh", "circ", "logistic", "circ_dist"],
    "Model to use for training.",
)
flags.DEFINE_enum(
    "optimizer", "adam", ["adam", "sgd", "rms"], "Optimizer to use for training."
)
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
flags.DEFINE_integer("start_epoch", 1, "Which epoch are we in?")
flags.DEFINE_string("resume", "", "File to resume from (if any)")
flags.DEFINE_boolean(
    "visualize_activations", False, "Should we visualize activations with visdom"
)


class PiecewiseLinear(namedtuple("PiecewiseLinear", ("batch_size", "knots", "vals"))):
    def step(self, optimizer, t):
        lr = np.interp([t], self.knots, self.vals)[0]
        for group in optimizer.param_groups:
            group["lr"] = lr / self.batch_size


def generate_poisson_trains(batch_size, num_trains, seq_length, freq):
    trains = np.random.rand(seq_length, batch_size, num_trains) < freq
    return torch.from_numpy(trains).float()


def add_luminance(images):
    return torch.cat(
        (
            images,
            torch.unsqueeze(
                0.2126 * images[0, :, :]
                + 0.7152 * images[1, :, :]
                + 0.0722 * images[2, :, :],
                0,
            ),
        ),
        0,
    )


def poisson_train(images, seq_length, rel_fmax=0.2):
    return (torch.rand(seq_length, *images.shape).float() < rel_fmax * images).float()


def signed_poisson_train(images, seq_length, rel_fmax=0.2):
    return (
        torch.sign(images)
        * (
            torch.rand(seq_length, *images.shape).float() < rel_fmax * torch.abs(images)
        ).float()
    )


class LIFConvNet(torch.nn.Module):
    def __init__(self, num_channels):
        super(LIFConvNet, self).__init__()

        if FLAGS.net == "convnet":
            dtype = torch.float
            self.rsnn = ConvNet(num_channels=num_channels, feature_size=32, dtype=dtype)
        elif FLAGS.net == "convnet4":
            self.rsnn = ConvNet4(num_channels=num_channels, feature_size=32)

    def forward(self, x):
        voltages = self.rsnn(x).permute(1, 0, 2)
        m, _ = torch.max(voltages, 0)
        log_p_y = torch.nn.functional.log_softmax(m, dim=1)
        return log_p_y


def train(
    model, device, train_loader, optimizer, epoch, lr_scheduler=None, writer=None
):
    model.train()
    losses = []
    train_batches = len(train_loader)
    step = train_batches * epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        if FLAGS.save_grads and batch_idx % FLAGS.grad_save_interval == 0:
            for idx, p in enumerate(model.parameters()):
                np.save(f"param-{idx}-{epoch}-{batch_idx}-grad.npy", p.grad.numpy())
                np.save(f"param-{idx}-{epoch}-{batch_idx}-data.npy", p.data.numpy())

        if lr_scheduler:
            lr_scheduler.step(optimizer, t=(epoch + batch_idx / train_batches))
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
            ts = np.arange(0, FLAGS.seq_length) * FLAGS.dt
            _, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
            axs = axs.reshape(-1)  # flatten
            for nrn in range(10):
                one_trace = model.voltages.detach().cpu().numpy()[:, 0, nrn]
                plt.sca(axs[nrn])
                plt.plot(ts, one_trace)
            plt.xlabel("Time [s]")
            plt.ylabel("Membrane Potential")
            plt.show()
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
            )  # get the index of the max log-probabilioty
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


def save(path, model, optimizer):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load(path, model, optimizer, device):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train(device=device)
    return model, optimizer


def main(args):
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

    constant_current_encoder = IFConstantCurrentEncoder(
        seq_length=FLAGS.seq_length, v_th=FLAGS.current_encoder_v_th
    )

    def polar_current_encoder(x):
        x_p, _ = constant_current_encoder(2 * torch.nn.functional.relu(x))
        x_m, _ = constant_current_encoder(2 * torch.nn.functional.relu(-x))
        return torch.cat((x_p, x_m), 1)

    def current_encoder(x):
        x, _ = constant_current_encoder(2 * x)
        return x

    def poisson_encoder(x):
        return poisson_train(x, seq_length=FLAGS.seq_length)

    def signed_poisson_encoder(x):
        return signed_poisson_train(x, seq_length=FLAGS.seq_length)

    def signed_current_encoder(x):
        z, _ = constant_current_encoder(torch.abs(x))
        return torch.sign(x) * z

    num_channels = 4

    if FLAGS.encoding == "poisson":
        encoder = poisson_encoder
    elif FLAGS.encoding == "constant":
        encoder = current_encoder
    elif FLAGS.encoding == "signed_poisson":
        encoder = signed_poisson_encoder
    elif FLAGS.encoding == "signed_constant":
        encoder = signed_current_encoder
    elif FLAGS.encoding == "constant_polar":
        encoder = polar_current_encoder
        num_channels = 2 * num_channels

    luminance_transforms = [
        add_luminance,
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465, 0.4816), (0.2023, 0.1994, 0.2010, 0.20013)
        ),
    ]

    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]
        + luminance_transforms
        + [encoder]
    )

    transform_test = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()] + luminance_transforms + [encoder]
    )

    kwargs = {"num_workers": 0, "pin_memory": True} if FLAGS.device == "cuda" else {}
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root=".", train=True, download=True, transform=transform_train
        ),
        batch_size=FLAGS.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root=".", train=False, transform=transform_test),
        batch_size=FLAGS.batch_size,
        **kwargs,
    )

    label = os.environ.get("SLURM_JOB_ID", str(uuid.uuid4()))
    if not FLAGS.prefix:
        rundir = f"runs/cifar10/{label}"
    else:
        rundir = f"runs/cifar10/{FLAGS.prefix}/{label}"

    os.makedirs(rundir, exist_ok=True)
    os.chdir(rundir)
    FLAGS.append_flags_into_file("flags.txt")

    model = LIFConvNet(num_channels=num_channels).to(device)

    print(model)

    if device == "cuda":
        model = torch.nn.DataParallel(model).to(device)

    if FLAGS.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=FLAGS.learning_rate,
            momentum=0.9,
            weight_decay=5e-4 * FLAGS.batch_size,
            nesterov=True,
        )
    elif FLAGS.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    elif FLAGS.optimizer == "rms":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=FLAGS.learning_rate)

    if FLAGS.only_output:
        optimizer = torch.optim.Adam(model.out.parameters(), lr=FLAGS.learning_rate)

    if FLAGS.resume:
        if os.path.isfile(FLAGS.resume):
            checkpoint = torch.load(FLAGS.resume)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])

    if FLAGS.learning_rate_schedule:
        lr_scheduler = PiecewiseLinear(
            FLAGS.batch_size, [0, 5, FLAGS.epochs], [0, 0.4, 0]
        )
    else:
        lr_scheduler = None

    training_losses = []
    mean_losses = []
    test_losses = []
    accuracies = []

    start = datetime.datetime.now()
    for epoch in range(FLAGS.start_epoch, FLAGS.start_epoch + FLAGS.epochs):
        training_loss, mean_loss = train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            lr_scheduler=lr_scheduler,
            writer=writer,
        )
        test_loss, accuracy = test(model, device, test_loader, epoch, writer=writer)

        training_losses += training_loss
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        if (epoch % FLAGS.model_save_interval == 0) and FLAGS.save_model:
            model_path = f"cifar10-{epoch}.pt"
            save(model_path, model, optimizer)

    stop = datetime.datetime.now()

    np.save("training_losses.npy", np.array(training_losses))
    np.save("mean_losses.npy", np.array(mean_losses))
    np.save("test_losses.npy", np.array(test_losses))
    np.save("accuracies.npy", np.array(accuracies))
    model_path = "cifar10-final.pt"
    save(model_path, model, optimizer)

    logging.info(f"output saved to {rundir}")
    logging.info(f"{start - stop}")
    if writer:
        writer.close()


if __name__ == "__main__":
    app.run(main)
