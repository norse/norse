from ..functional.lif import (
    LIFState,
    LIFFeedForwardState,
    LIFParameters,
    lif_step,
    lif_feed_forward_step,
    lif_current_encoder,
)

import torch
import time
import numpy as np


def lif_benchmark(
        input_features=10,
        output_features=10,
        n_time_steps=100,
        batch_size=16,
        device="cpu"
):
    input_spikes = torch.randn(
        (n_time_steps, batch_size, input_features)).to(device)
    iw = torch.randn(output_features, input_features).to(device)
    rw = torch.randn(output_features, output_features).to(device)
    T = n_time_steps
    s = LIFState(
        z=torch.zeros(batch_size, output_features).to(device),
        v=torch.zeros(batch_size, output_features).to(device),
        i=torch.zeros(batch_size, output_features).to(device)
    )
    p = LIFParameters(
        alpha=100.0,
        method="heaviside"
    )

    start = time.time()
    for ts in range(T):
        z, s = lif_step(
            input=input_spikes[ts, :],
            s=s,
            input_weights=iw,
            recurrent_weights=rw,
            p=p,
            dt=0.001
        )

    end = time.time()
    dt = (end - start) / T
    return dt


def lif_feed_forward_benchmark(
        input_features=10,
        output_features=10,
        n_time_steps=100,
        batch_size=16,
        device="cpu"
):
    input_spikes = torch.randn(
        (n_time_steps, batch_size, input_features)).to(device)
    fc = torch.nn.Linear(input_features, output_features,
                         bias=False).to(device)
    T = n_time_steps
    s = LIFFeedForwardState(
        v=torch.zeros(batch_size, output_features).to(device),
        i=torch.zeros(batch_size, output_features).to(device)
    )
    p = LIFParameters(
        alpha=100.0,
        method="heaviside"
    )

    start = time.time()
    for ts in range(T):
        x = fc(input_spikes[ts, :])
        z, s = lif_feed_forward_step(
            input=x,
            s=s,
            p=p,
            dt=0.01
        )

    end = time.time()
    dt = (end - start) / T
    return dt
