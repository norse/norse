from typing import List, NamedTuple

import numpy as np


class BenchmarkConfig(NamedTuple):
    """
    Benchmark configurations for the benchmarking setup
    """

    batch_size: int
    device: str
    dt: float
    label: str
    runs: int
    profile: bool
    sequence_length: int
    start: int
    stop: int
    step: int


class BenchmarkParameters(NamedTuple):
    """
    Benchmark parameters used as input for networks
    """

    device: str
    dt: float
    features: int
    batch_size: int
    sequence_length: int


class BenchmarkData(NamedTuple):
    """
    Result from a benchmark run N number of times
    """

    config: BenchmarkConfig
    durations: np.ndarray
    parameters: BenchmarkParameters
