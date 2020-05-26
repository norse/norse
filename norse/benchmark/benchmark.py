import numpy as np


def benchmark(run, n_runs=100, **kwargs):
    dts = []
    for _ in range(n_runs):
        dt = run(**kwargs)
        dts.append(dt)

    return np.stack(dts)
