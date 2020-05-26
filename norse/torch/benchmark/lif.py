from norse.torch.functional.lif import (
    LIFState,
    LIFFeedForwardState,
    LIFParameters,
    lif_step,
    lif_feed_forward_step,
)
from norse.torch.module.encode import PoissonEncoder

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_neurons", 1000, "Number of neurons to simulate")
flags.DEFINE_integer("start", 250, "Start of the number of inputs to sweep")
flags.DEFINE_integer("step", 250, "Steps in which to sweep over the number of inputs")
flags.DEFINE_integer("stop", 10000, "Number of inputs to sweep to")
flags.DEFINE_integer("sequence_length", 1000, "Number of timesteps to simulate")
flags.DEFINE_string("device", "cpu", "Device to use [cpu, cuda]")


import torch
import time
import numpy as np
import pandas as pd


def lif_benchmark(
    input_features=10,
    output_features=10,
    n_time_steps=100,
    batch_size=16,
    input_spikes=torch.zeros((100, 16, 10)),
):
    iw = torch.randn(output_features, input_features)
    rw = torch.randn(output_features, output_features)
    T = n_time_steps
    s = LIFState(
        z=torch.zeros(batch_size, output_features),
        v=torch.zeros(batch_size, output_features),
        i=torch.zeros(batch_size, output_features),
    )
    p = LIFParameters(alpha=100.0, method="heaviside")

    start = time.time()
    for ts in range(T):
        _, s = lif_step(
            input_tensor=input_spikes[ts, :],
            state=s,
            input_weights=iw,
            recurrent_weights=rw,
            parameters=p,
            dt=0.001,
        )

    end = time.time()
    dt = (end - start) / T
    return dt


def lif_feed_forward_benchmark(
    input_features=10, output_features=10, n_time_steps=100, batch_size=16
):
    fc = torch.nn.Linear(input_features, output_features, bias=False).to(FLAGS.device)
    T = n_time_steps
    s = LIFFeedForwardState(
        v=torch.zeros(batch_size, output_features).to(FLAGS.device),
        i=torch.zeros(batch_size, output_features).to(FLAGS.device),
    )
    p = LIFParameters(alpha=100.0, method="heaviside")
    input_spikes = PoissonEncoder(n_time_steps)(
        0.3 * torch.ones(batch_size, input_features, device = FLAGS.device)
    ).to(FLAGS.device)
    start = time.time()

    spikes = []
    for ts in range(T):
        x = fc(input_spikes[ts, :])
        z, s = lif_feed_forward_step(input_tensor=x, state=s, parameters=p, dt=0.001)
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
        "dt": duration / T,
        "time_steps": T,
        "device": FLAGS.device,
    }
    return result


def main(argv):

    batch_sizes = [2 ** i for i in range(10)]
    results = []

    for batch_size in batch_sizes:
        for n_inputs in range(FLAGS.start, FLAGS.stop, FLAGS.step):
            result = lif_feed_forward_benchmark(
                output_features=n_inputs, input_features=n_inputs, batch_size=batch_size
            )
            logging.info(result)
            results += [result]

    data = pd.DataFrame(results)
    data.to_csv("results/benchmark.csv")


if __name__ == "__main__":
    app.run(main)
