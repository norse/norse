from absl import app
from absl import flags
from absl import logging

import csv, time, torch

from bindsnet.network.nodes import LIFNodes
from bindsnet.encoding import PoissonEncoder

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "batches",
    10,
    "Number of batch sizes to simulate in base-2 (2 = [1, 2] and 4 = [1, 2, 4, 8] for instance",
)
flags.DEFINE_integer("num_neurons", 1000, "Number of neurons to simulate")
flags.DEFINE_integer("start", 250, "Start of the number of inputs to sweep")
flags.DEFINE_integer("step", 250, "Steps in which to sweep over the number of inputs")
flags.DEFINE_integer("stop", 10000, "Number of inputs to sweep to")
flags.DEFINE_integer("sequence_length", 1000, "Number of timesteps to simulate")
flags.DEFINE_float("dt", 0.01, "Simulation timestep")
flags.DEFINE_string("device", "cpu", "Device to use [cpu, cuda]")


def lif_feed_forward_benchmark(
    input_features=10,
    output_features=10,
    n_time_steps=100,
    batch_size=16,
    dt=0.001,
    device="cpu",
):
    fc = torch.nn.Linear(input_features, output_features, bias=False).to(device)
    T = n_time_steps
    input_spikes = (
        PoissonEncoder(time=n_time_steps, dt=dt)(
            0.3 * torch.ones(batch_size, input_features)
        )
        .to(device)
        .float()
    )
    model = LIFNodes(
        n=input_features * output_features,
        shape=(input_features, output_features),
        learning=False,
        dt=dt,
    ).to(device)
    model.dt = dt

    start = time.time()

    spikes = []
    for ts in range(T):
        x = fc(input_spikes[ts, :])
        model(x)
        spikes += [model.s]

    spikes = torch.stack(spikes)
    end = time.time()
    duration = end - start

    result = {
        "label": "lif_feed_forward",
        "input_features": input_features,
        "output_features": output_features,
        "batch_size": batch_size,
        "duration": duration,
        "dt": dt,
        "time_steps": T,
        "device": device,
    }
    return result


def main(argv):

    batch_sizes = [2 ** i for i in range(FLAGS.batches)]
    results = []

    for batch_size in batch_sizes:
        for n_inputs in range(FLAGS.start, FLAGS.stop, FLAGS.step):
            result = lif_feed_forward_benchmark(
                output_features=n_inputs,
                input_features=n_inputs,
                batch_size=batch_size,
                dt=FLAGS.dt,
                device=FLAGS.device,
            )
            logging.info(result)
            results += [result]

    timestamp = time.strftime("%Y-%M-%d-%H-%M-%S")
    filename = f"bindsnet-lif-{timestamp}.csv"
    with open(filename, "w") as f:
        for index, result in enumerate(results):
            w = csv.DictWriter(f, result.keys())
            if index == 0:
                w.writeheader()
            w.writerow(result)


if __name__ == "__main__":
    app.run(main)
