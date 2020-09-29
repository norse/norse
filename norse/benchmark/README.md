This benchmarking package provides preliminary comparisons to 
other spike-based deep learning approaches.
We plan to extend it with more benchmarks in the immediate future.
On a longer timescale, we hope to document the improved quality and speed
of Norse by comparing future performance with past releases.

## LIF benchmark comparison with Norse, BindsNET, and GeNN
BindsNET and GeNN are two of the closest competitors in the SNN space - at
least in terms of performance.
The graph below shows a benchmark between Norse, BindsNET, and GeNN simulating
poisson encoded input to a single layer of LIF neurons. The simulation ran on a
Intel i7-5930K 6-core CPU machine with a NVIDIA GTX Titan X card (12 Gb RAM) for 1000 
timesteps with a time-delta of 0.001 seconds and a batch-size of 32.
Each line indicates the mean running time out of 20 repeated runs and the coloured
area indicates where ~95% of the runtimes fell.

BindsNET ran out of memory in > 4500 neuron layers, indicated by the orange circle.

The benchmark indicates that for single-layer LIF neurons of size < 4500, Norse outperforms both competitors.
The overtaking of GeNN is likely due to the fact that they do not use matrix multiplications, which scales quadratically with size in Norse and BindsNET.
This is fairly straight-forward to overcome and that we will work in 
going forward, although it is rare to see that many neurons in a single layer.

![](lif_benchmark.png)