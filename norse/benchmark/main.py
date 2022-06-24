from argparse import ArgumentParser
from functools import partial
import logging
from pathlib import Path
from typing import Callable

import numpy as np

# pytype: disable=import-error
import pandas as pd

# pytype: enable=import-error

import gc

# pytype: disable=import-error
from .benchmark import BenchmarkConfig, BenchmarkData, BenchmarkParameters

# pytype: enable=import-error


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


def main(args):
    # pytype: disable=import-error
    if args.bindsnet:
        import bindsnet_lif

        run_benchmark(
            args, bindsnet_lif.lif_feed_forward_benchmark, label="BindsNET_lif"
        )
    if args.genn:
        import genn_lif

        run_benchmark(args, genn_lif.lif_feed_forward_benchmark, label="GeNN_lif")
    if args.norse:
        import norse
        from . import norse_lif

        if args.profile:
            import torch.autograd.profiler as profiler

            with profiler.profile(
                profile_memory=True, use_cuda=(args.device == "cuda")
            ) as prof:
                run_benchmark(
                    args,
                    norse_lif.lif_feed_forward_benchmark,
                    label=f"Norse v{norse.__version__} lif",
                )
            prof.export_chrome_trace("trace.json")
        else:
            run_benchmark(
                args,
                norse_lif.lif_feed_forward_benchmark,
                label=f"Norse v{norse.__version__} lif",
            )
    # pytype: enable=import-error


def run_benchmark(args, function, label):
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        device=args.device,
        dt=args.dt,
        label=label,
        runs=args.runs,
        sequence_length=args.sequence_length,
        start=args.start,
        stop=args.stop,
        step=args.step,
        profile=args.profile,
    )

    collector = partial(collect, label=label)
    results = benchmark(function, collector, config)

    filename = f"benchmark_results.csv"
    is_file = Path(filename).is_file()
    pd.DataFrame(results).to_csv(filename, mode="a", index=False, header=not is_file)


if __name__ == "__main__":
    parser = ArgumentParser("SNN library benchmarks")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of data points per batch",
    )
    parser.add_argument(
        "--start", type=int, default=250, help="Start of the number of inputs to sweep"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=250,
        help="Steps in which to sweep over the number of inputs",
    )
    parser.add_argument(
        "--stop", type=int, default=5001, help="Number of inputs to sweep to"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1000,
        help="Number of timesteps to simulate",
    )
    parser.add_argument("--dt", type=float, default=0.001, help="Simulation timestep")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use [cpu, cuda]",
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of runs per simulation step"
    )
    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="Profile Norse benchmark? (Only works for Norse)",
    )
    parser.add_argument(
        "--bindsnet",
        default=False,
        action="store_true",
        help="Benchmark Bindsnet?",
    )
    parser.add_argument(
        "--genn", default=False, action="store_true", help="Benchmark GeNN?"
    )
    parser.add_argument(
        "--norse",
        default=False,
        action="store_true",
        help="Benchmark Norse?",
    )
    args = parser.parse_args()
    main(args)
