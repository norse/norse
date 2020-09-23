import numpy as np
import time
import sys
from subprocess import run, STDOUT
from multiprocessing.shared_memory import ShareableList
from multiprocessing.managers import SharedMemoryManager

from pygenn.genn_model import GeNNModel
from pygenn.genn_wrapper import NO_DELAY
from pygenn import genn_wrapper

from benchmark import BenchmarkParameters


def lif_feed_forward_benchmark(parameters: BenchmarkParameters):
    shared = SharedMemoryManager()
    shared.start()
    params = list(parameters._asdict().values())
    shared_list = shared.ShareableList(params)

    run(["python3", __file__, shared_list.shm.name], stderr=STDOUT)
    print(shared_list)
    duration = shared_list[0]
    shared.shutdown()
    return duration

if __name__ == "__main__":
    # Assume we're running the genn benchmark and draw configs from the shared memory
    parameter_list = ShareableList(sequence=None, name=sys.argv[1])
    parameters = BenchmarkParameters(*parameter_list)

    try:
        T = int(parameters.sequence_length / parameters.dt)

        model = GeNNModel("float", "pygenn")
        model.dT = parameters.dt

        for i in range(parameters.batch_size):
            ones = np.ones(parameters.features)
            model.add_neuron_population(
                f"PoissonNew{i}", parameters.features, "PoissonNew", {"rate": 0.3}, {"timeStepToSpike": 1}
            )
            # From https://neuralensemble.org/docs/PyNN/reference/neuronmodels.html#pyNN.standardmodels.cells.IF_curr_exp
            lif_params = { 
                "C": 1.0,
                "TauM": 20.0,
                "Vrest": -65.0,
                "Vreset": -65.0,
                "Vthresh": -50.0,
                "Ioffset": 0.0,
                "TauRefrac": 0.1
            }
            lif_vars = {"V": -65.0 * ones, "RefracTime": 0.0 * ones}
            model.add_neuron_population(
                f"LIF{i}", parameters.features, "LIF", lif_params, lif_vars
            )
            # From https://github.com/genn-team/genn/blob/master/userproject/PoissonIzh_project/model/PoissonIzh.cc#L93
            model.add_synapse_population(f"PL{i}", "DENSE_INDIVIDUALG", NO_DELAY, f"PoissonNew{i}", f"LIF{i}", "StaticPulse", {}, {"g": 1}, {}, {}, "DeltaCurr", {}, {})

        model.build()
        model.load()

        # Run simulation
        start = time.time()
        for _ in range(T):
            model.step_time()
            for i in range(parameters.batch_size):
                model.pull_state_from_device(f'LIF{i}')
        end = time.time()
        
        parameter_list[0] = end - start
    finally:
        parameter_list.unbind()