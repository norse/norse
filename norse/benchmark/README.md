This benchmarking package provides preliminary comparisons to 
other spike-based deep learning approaches.
We plan to extend it with more benchmarks in the immediate future.
On a longer timescale, we hope to document the improved quality and speed
of Norse by comparing future performance with past releases.

## LIF benchmark comparison with Norse, PySNN, and BindsNET
BindsNET and PySNN are two of the closest competitors in the SNN space.
The graph below shows a benchmark between Norse, BindsNET, and PySNN simulating
poisson encoded input to a single layer of LIF neurons. The simulation ran on a
Intel i7-5930K 6-core CPU machine with a NVIDIA GTX Titan X card for 1000 timesteps 
with a time-delta of 0.001 seconds and a batch-size of 16.
Each run was repeated 100 times and the area where ~95% of the runtimes
fell is grayed out in the graph to illustrate the variance.

The benchmark indicates that for single-layer LIF neurons, Norse outperforms
both competitors.

![](lif_benchmark.png)