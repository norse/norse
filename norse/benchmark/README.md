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

The benchmark indicates that for a single layer of less than 4500 LIF neurons, Norse outperforms both competitors.
For larger numbers of neurons per layer, GeNN is likely faster due to the fact that they do not use matrix multiplications, but event based processing. 
This scales with the number of events to be processes as opposed to quadratically in the case of Norse and BindsNET.
This limitation is straight-forward to overcome, if one uses the fact that only (sparse) bit-vectors need to processed, 
although it is rare to see that many neurons in a single layer in machine-learning tasks.

![](lif_benchmark.png)
