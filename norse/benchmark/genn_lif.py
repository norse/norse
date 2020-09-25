import numpy as np
import time
from pygenn.genn_model import GeNNModel
from pygenn.genn_wrapper import NO_DELAY
from pygenn import genn_wrapper

from benchmark import BenchmarkParameters


def lif_feed_forward_benchmark(parameters: BenchmarkParameters):
    T = parameters.sequence_length / parameters.dt

    model = GeNNModel("float", "pygenn", backend_log_level=genn_wrapper.debug)
    model.dT = parameters.dt

    run(["python3", __file__, shared_list.shm.name], stderr=STDOUT)
    duration = shared_list[0]
    shared_list.shm.close()
    shared.shutdown()
    return duration

if __name__ == "__main__":
    # Assume we're running the genn benchmark and draw configs from the shared memory
    parameter_list = ShareableList(sequence=None, name=sys.argv[1])
    parameters = BenchmarkParameters(*parameter_list)

    model = GeNNModel("float", "pygenn")
    model.dT = parameters.dt
    np.random.seed(0)

    layers = []
    for i in range(parameters.batch_size):
        ones = np.ones(parameters.features)
        # Note: weights, parameters and poisson rate are magic numbers that seem to generate reasonable spike activity
        weights = np.random.rand(parameters.features, parameters.features).flatten() * 8
        model.add_neuron_population(
            f"PoissonNew{i}", parameters.features, "PoissonNew", {"rate": 100}, {"timeStepToSpike": 1}
        )
        # From https://neuralensemble.org/docs/PyNN/reference/neuronmodels.html#pyNN.standardmodels.cells.IF_curr_exp
        lif_params = { 
            "C": 1.0,
            "TauM": 20.0,
            "Vrest": 0.0,
            "Vreset": 0.0,
            "Vthresh": 1.0,
            "Ioffset": 0.0,
            "TauRefrac": 0.1
        }
        lif_vars = {"V": 0.0 * ones, "RefracTime": 0.0 * ones}
        layer = model.add_neuron_population(
            f"LIF{i}", parameters.features, "LIF", lif_params, lif_vars
        )
        layers.append(layer)
        # From https://github.com/genn-team/genn/blob/master/userproject/PoissonIzh_project/model/PoissonIzh.cc#L93
        model.add_synapse_population(f"PL{i}", "DENSE_INDIVIDUALG", NO_DELAY, f"PoissonNew{i}", f"LIF{i}", 
                                     "StaticPulse", {}, {"g": weights}, {}, {}, "DeltaCurr", {}, {})

    model.build()
    model.load()

    # Run simulation
    start = time.time()
    layer_spikes = []
    for _ in range(parameters.sequence_length):
        model.step_time()
        timestep_spikes = []
        # From https://github.com/neworderofjamie/pygenn_ml_tutorial/blob/master/tutorial_1.py
        for layer in layers:
            layer.pull_current_spikes_from_device()
            timestep_spikes.append(np.copy(layer.current_spikes))
        layer_spikes.append(timestep_spikes)
    end = time.time()
    
    parameter_list[0] = end - start
    parameter_list[1] = len(np.array(layer_spikes).flatten())
    parameter_list.shm.close()
