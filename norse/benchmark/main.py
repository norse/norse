from functools import partial
from typing import Callable

from absl import app
from absl import flags
from absl import logging

import numpy as np
import pandas as pd
import time

from benchmark import *

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "batches",
    10,
    "Number of batch sizes to simulate in base-2 (2 = [1, 2] and 4 = [1, 2, 4, 8] for instance)",
)
flags.DEFINE_integer("start", 250, "Start of the number of inputs to sweep")
flags.DEFINE_integer("step", 250, "Steps in which to sweep over the number of inputs")
flags.DEFINE_integer("stop", 5000, "Number of inputs to sweep to")
flags.DEFINE_integer("sequence_length", 1000, "Number of timesteps to simulate")
flags.DEFINE_float("dt", 0.001, "Simulation timestep")
flags.DEFINE_string("device", "cpu", "Device to use [cpu, cuda]")
flags.DEFINE_integer("runs", 100, "Number of runs per simulation step")
flags.DEFINE_bool("profile", False, "Profile Norse benchmark? (Only works for Norse)")

flags.DEFINE_bool("bindsnet", True, "Benchmark Bindsnet?")
flags.DEFINE_bool("genn", True, "Benchmark GeNN?")
flags.DEFINE_bool("norse", True, "Benchmark Norse?")
flags.DEFINE_bool("pysnn", True, "Benchmark PySNN?")


def benchmark(
    model: Callable[[BenchmarkParameters], float],
    collector: Callable[[BenchmarkData], dict],
    config: BenchmarkConfig,
):
    """
    Benchmarks a model with the given configurations
    """
    results = []
    for batch_number in config.batch_sizes:
        for features in range(config.start, config.stop, config.step):
            parameters = BenchmarkParameters(
                device=config.device,
                dt=config.dt,
                features=features,
                batch_size=batch_number,
                sequence_length=config.sequence_length,
            )

            durations = []
            try:
                for _ in range(config.runs):
                    duration = model(parameters)
                    durations.append(duration)
            except RuntimeError as e:
                message = (
                    f"RuntimeError when running benchmark {config} {parameters}: {e}"
                )
                logging.error(message)

            data = BenchmarkData(
                config=config,
                durations=np.array(durations),
                parameters=parameters,
            )
            result = collector(data)

            logging.info(result)
            results += [result]

    return results


def collect(data: BenchmarkData, label: str) -> dict:
    return {
        "label": label,
        "input_features": data.parameters.features,
        "output_features": data.parameters.features,
        "batch_size": data.parameters.batch_size,
        "run": len(data.durations),
        "duration_mean": data.durations.mean(),
        "duration_std": data.durations.std(),
        "dt": data.parameters.dt,
        "time_steps": data.parameters.sequence_length,
        "device": data.config.device,
    }


def main(argv):
    if FLAGS.bindsnet:
        import bindsnet_lif

        run_benchmark(bindsnet_lif.lif_feed_forward_benchmark, "bindsnet_lif")
    if FLAGS.genn:
        import genn_lif

        run_benchmark(genn_lif.lif_feed_forward_benchmark, "genn_lif")
    if FLAGS.norse:
        import norse_lif

        if FLAGS.profile:
            import torch.autograd.profiler as profiler
            with profiler.profile(profile_memory=True, use_cuda=(FLAGS.device == 'cuda')) as prof:
                run_benchmark(norse_lif.lif_feed_forward_benchmark, "norse_lif")
            prof.export_chrome_trace("trace.json")
        else:
            run_benchmark(norse_lif.lif_feed_forward_benchmark, "norse_lif")
    if FLAGS.pysnn:
        import pysnn_lif

        run_benchmark(pysnn_lif.lif_feed_forward_benchmark, "pysnn_lif")


def run_benchmark(function, label):
    config = BenchmarkConfig(
        batch_sizes=[2 ** i for i in range(FLAGS.batches)],
        device=FLAGS.device,
        dt=FLAGS.dt,
        label=label,
        runs=FLAGS.runs,
        sequence_length=FLAGS.sequence_length,
        start=FLAGS.start,
        stop=FLAGS.stop,
        step=FLAGS.step,
        profile=FLAGS.profile
    )

    collector = partial(collect, label=label)
    results = benchmark(function, collector, config)

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{timestamp}-{label}.csv"
    pd.DataFrame(results).to_csv(filename)


if __name__ == "__main__":
    app.run(main)
