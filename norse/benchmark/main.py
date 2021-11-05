from functools import partial
from typing import Callable

from absl import app
from absl import flags
from absl import logging

import numpy as np

# pytype: disable=import-error
import pandas as pd
# pytype: enable=import-error

import time
import gc

# pytype: disable=import-error
from benchmark import BenchmarkConfig, BenchmarkData, BenchmarkParameters
# pytype: enable=import-error

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "batch_size",
    32,
    "Number of data points per batch",
)
flags.DEFINE_integer("start", 250, "Start of the number of inputs to sweep")
flags.DEFINE_integer("step", 250, "Steps in which to sweep over the number of inputs")
flags.DEFINE_integer("stop", 5001, "Number of inputs to sweep to")
flags.DEFINE_integer("sequence_length", 1000, "Number of timesteps to simulate")
flags.DEFINE_float("dt", 0.001, "Simulation timestep")
flags.DEFINE_enum("device", "cuda", ["cuda", "cpu"], "Device to use [cpu, cuda]")
flags.DEFINE_integer("runs", 5, "Number of runs per simulation step")
flags.DEFINE_bool("profile", False, "Profile Norse benchmark? (Only works for Norse)")

flags.DEFINE_bool("bindsnet", False, "Benchmark Bindsnet?")
flags.DEFINE_bool("genn", False, "Benchmark GeNN?")
flags.DEFINE_bool("norse", False, "Benchmark Norse?")


def benchmark(
    model: Callable[[BenchmarkParameters], float],
    collector: Callable[[BenchmarkData], dict],
    config: BenchmarkConfig,
):
    """
    Benchmarks a model with the given configurations
    """
    results = []
    for features in range(config.start, config.stop, config.step):
        parameters = BenchmarkParameters(
            device=config.device,
            dt=config.dt,
            features=features,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
        )

        durations = []
        try:
            for _ in range(config.runs):
                duration = model(parameters)
                durations.append(duration)
                # Clean up by GC and empty cache
                # Thanks to https://github.com/BindsNET/bindsnet/issues/425#issuecomment-721780231
                gc.collect()
                try:
                    import torch

                    with torch.no_grad():
                        torch.cuda.empty_cache()
                except:
                    pass
        except KeyboardInterrupt:
            raise KeyboardInterrupt()
        except RuntimeError as e:
            message = f"RuntimeError when running benchmark {config} {parameters}: {e}"
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
    # pytype: disable=import-error
    if FLAGS.bindsnet:
        from norse.benchmark import bindsnet_lif

        run_benchmark(bindsnet_lif.lif_feed_forward_benchmark, "BindsNET_lif")
    if FLAGS.genn:
        from norse.benchmark import genn_lif

        run_benchmark(genn_lif.lif_feed_forward_benchmark, "GeNN_lif")
    if FLAGS.norse:
        from norse.benchmark import norse_lif

        if FLAGS.profile:
            import torch.autograd.profiler as profiler

            with profiler.profile(
                profile_memory=True, use_cuda=(FLAGS.device == "cuda")
            ) as prof:
                run_benchmark(norse_lif.lif_feed_forward_benchmark, "Norse_lif")
            prof.export_chrome_trace("trace.json")
        else:
            run_benchmark(norse_lif.lif_feed_forward_benchmark, "Norse_lif")
    # pytype: enable=import-error


def run_benchmark(function, label):
    config = BenchmarkConfig(
        batch_size=FLAGS.batch_size,
        device=FLAGS.device,
        dt=FLAGS.dt,
        label=label,
        runs=FLAGS.runs,
        sequence_length=FLAGS.sequence_length,
        start=FLAGS.start,
        stop=FLAGS.stop,
        step=FLAGS.step,
        profile=FLAGS.profile,
    )

    collector = partial(collect, label=label)
    results = benchmark(function, collector, config)

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{timestamp}-{label}.csv"
    pd.DataFrame(results).to_csv(filename)


if __name__ == "__main__":
    app.run(main)
