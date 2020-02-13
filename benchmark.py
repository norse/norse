from norse.torch.benchmark.lif import lif_feed_forward_benchmark, lif_benchmark
from norse.torch.benchmark.benchmark import benchmark

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

if __name__ == "__main__":
    n_time_steps = 1000
    feature_sizes = range(250, 10001, 250)
    batch_size = 256

    timings = []

    for features in feature_sizes:
        input_spikes = torch.from_numpy((np.random.uniform(size=(n_time_steps, batch_size, features)) > 0.5)).float()
        dts = benchmark(
            lif_feed_forward_benchmark,
            n_runs=1,
            n_time_steps=n_time_steps,
            input_features=features,
            output_features=features,
            input_spikes=input_spikes,
            batch_size=batch_size
        )
        timing = {
            'features': features,
            'mean': n_time_steps * np.mean(dts) / batch_size,
            'std': np.std(dts),
            'min': np.min(dts),
            'max': np.max(dts)
        }
        print(timing)
        timings.append(timing)

    with open('benchmark.pt', 'wb') as f:
        pickle.dump(timings, f)
