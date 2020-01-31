"""
This module implements standard benchmarks for norse.
"""
import numpy as np


def benchmark(function, parameters, value_range, device="cpu", n_runs=10, aggregration_function=np.mean):
    """
    Benchmarks a given function where the list of parameters 
    is set to the given value_range, iterated over the length of the range.
    Returns a dictionary of values and associated aggregated value.

    O(n_runs * len(range))
    """
    results = {}
    for i in value_range:
        args = {parameter: i for parameter in parameters}
        args["device"] = device
        result_list = run(function, n_runs, **args)
        results[i] = aggregration_function(result_list)
    return results


def run(function, n_runs=10, **kwargs):
    """
    A simple function that executes a function n_runs
    times with the given key-value parameters.
    """
    dts = []
    for _ in range(n_runs):
        dt = function(**kwargs)
        dts.append(dt)

    return np.stack(dts)
