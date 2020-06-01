import csv
import time
import torch

from absl import app
from absl import flags
from absl import logging

from pysnn.connection import Linear
from pysnn.encoding import PoissonEncoder
from pysnn.neuron import LIFNeuron, Input
from pysnn.network import SNNNetwork

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "batches",
    10,
    "Number of batch sizes to simulate in base-2 (2 = [1, 2] and 4 = [1, 2, 4, 8] for instance",
)
flags.DEFINE_integer("start", 250, "Start of the number of inputs to sweep")
flags.DEFINE_integer("step", 250, "Steps in which to sweep over the number of inputs")
flags.DEFINE_integer("stop", 10000, "Number of inputs to sweep to")
flags.DEFINE_integer("sequence_length", 1000, "Number of timesteps to simulate")
flags.DEFINE_float("dt", 0.001, "Simulation timestep")
flags.DEFINE_string("device", "cpu", "Device to use [cpu, cuda]")


def lif_feed_forward_benchmark(
    input_features=10,
    output_features=10,
    n_time_steps=100,
    batch_size=16,
    dt=0.001,
    device="cpu",
):
    T = int(dt * n_time_steps)

    # Default parameters from Norse neuron model
    tau_syn = torch.as_tensor(5e-3)
    tau_mem = torch.as_tensor(1e-2)
    v_th = torch.as_tensor(1.0)
    v_reset = torch.as_tensor(0.0)

    input_layer = Input(
        (batch_size, 1, input_features), dt=dt, alpha_t=1.0, tau_t=1.0
    ).to(device)
    linear_layer = Linear(
        in_features=input_features,
        out_features=output_features,
        batch_size=batch_size,
        dt=dt,
        delay=0,
    ).to(device)
    lif_layer = LIFNeuron(
        cells_shape=(batch_size, 1, output_features),
        thresh=v_th,
        v_rest=v_reset,
        alpha_v=1.0,
        alpha_t=1.0,
        dt=dt,
        duration_refrac=0.001,
        tau_v=tau_syn,
        tau_t=tau_mem,
        update_type="exponential",
    ).to(device)

    input_spikes = (
        PoissonEncoder(duration=T, dt=dt)(0.3 * torch.ones(batch_size, input_features))
        .reshape(batch_size, 1, input_features, n_time_steps)
        .to(device)
    )

    start = time.time()
    spikes = []
    for ts in range(n_time_steps):
        z, t = input_layer(input_spikes[:, :, :, ts])
        z, _ = linear_layer(z, t)
        z, _ = lif_layer(z)
        spikes += [z]

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

    try:
        for batch_size in batch_sizes:
            for n_inputs in range(FLAGS.start, FLAGS.stop, FLAGS.step):
                result = lif_feed_forward_benchmark(
                    output_features=n_inputs,
                    input_features=n_inputs,
                    batch_size=batch_size,
                    n_time_steps=FLAGS.sequence_length,
                    dt=FLAGS.dt,
                    device=FLAGS.device,
                )
                logging.info(result)
                results += [result]
    except RuntimeError:
        logging.error("RuntimeError when running benchmark")

    timestamp = time.strftime("%Y-%M-%d-%H-%M-%S")
    filename = f"pysnn-lif-{timestamp}.csv"
    with open(filename, "w") as f:
        for index, result in enumerate(results):
            w = csv.DictWriter(f, result.keys())
            if index == 0:
                w.writeheader()
            w.writerow(result)


if __name__ == "__main__":
    app.run(main)
