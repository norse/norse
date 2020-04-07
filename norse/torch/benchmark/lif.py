from ..functional.lif import (
    LIFState,
    LIFFeedForwardState,
    LIFParameters,
    lif_step,
    lif_feed_forward_step,
)

import torch
import time
import numpy as np


def lif_benchmark(
    input_features=10,
    output_features=10,
    n_time_steps=100,
    batch_size=16,
    input_spikes=np.zeros((100, 16, 10)),
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
        z, s = lif_step(
            input=input_spikes[ts, :],
            s=s,
            input_weights=iw,
            recurrent_weights=rw,
            p=p,
            dt=0.001,
        )

    end = time.time()
    dt = (end - start) / T
    return dt


def lif_feed_forward_benchmark(
    input_features=10,
    output_features=10,
    n_time_steps=100,
    batch_size=16,
    input_spikes=np.zeros((100, 16, 10)),
):
    fc = torch.nn.Linear(input_features, output_features, bias=False)
    T = n_time_steps
    s = LIFFeedForwardState(
        v=torch.zeros(batch_size, output_features),
        i=torch.zeros(batch_size, output_features),
    )
    p = LIFParameters(alpha=100.0, method="heaviside")

    start = time.time()
    for ts in range(T):
        x = fc(input_spikes[ts, :])
        z, s = lif_feed_forward_step(input=x, s=s, p=p, dt=0.01)

    end = time.time()
    dt = (end - start) / T
    return dt
