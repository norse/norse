from argparse import ArgumentParser
from functools import partial
import logging
from pathlib import Path
from typing import Callable
import tqdm

import numpy as np

# pytype: disable=import-error
import pandas as pd

# pytype: enable=import-error

import gc

# pytype: disable=import-error
from .benchmark import BenchmarkConfig, BenchmarkData, BenchmarkParameters
from .lif_box import main as lif_box_main
from .lif import main as lif_main

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
    for features in tqdm.tqdm(
        range(config.start, config.stop, config.step), desc=f"{config.label} - Features"
    ):
        parameters = BenchmarkParameters(
            device=config.device,
            dt=config.dt,
            features=features,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
        )

        durations = []
        try:
            for _ in tqdm.tqdm(range(config.runs), desc="Runs", leave=False):
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
    subparsers = parser.add_subparsers(help="Task types", required=True)
    lif_main.init_parser(subparsers.add_parser("lif"))
    lif_box_main.init_parser(subparsers.add_parser("lif_box"))
    args = parser.parse_args()
    args.func(args, run_benchmark)
