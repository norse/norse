import numpy as np
import time
from pygenn.genn_model import GeNNModel
from pygenn.genn_wrapper import NO_DELAY

from benchmark import BenchmarkParameters


def lif_feed_forward_benchmark(parameters: BenchmarkParameters):
    T = parameters.sequence_length / parameters.dt

    model = GeNNModel("float", "pygenn")
    model.dT = parameters.dt

    N = parameters.features * parameters.batch_size
    ones = np.ones(N)
    model.add_neuron_population(
        "PoissonNew", N, "PoissonNew", {"rate": 0.3}, {"timeStepToSpike": 1}
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
        "LIF", parameters.features * parameters.batch_size, "LIF", lif_params, lif_vars
    )
    # From https://github.com/genn-team/genn/blob/master/userproject/PoissonIzh_project/model/PoissonIzh.cc#L93
    model.add_synapse_population("PL", "DENSE_INDIVIDUALG", NO_DELAY, "PoissonNew", "LIF", "StaticPulse", {}, {"g": 1}, {}, {}, "DeltaCurr", {}, {})

    model.build()
    model.load()

    # Run simulation
    start = time.time()
    while model.t < T:
        model.step_time()
        model.pull_state_from_device('LIF')
    end = time.time()
    print(model.t)
    return end - start
    # network = Network(batch_size=parameters.batch_size, dt=parameters.dt)

    # network.add_layer(Input(n=parameters.features), name="Input")
    # network.add_layer(LIFNodes(n=parameters.features), name="Neurons")
    # network.add_connection(
    #     Connection(source=network.layers["Input"], target=network.layers["Neurons"]),
    #     source="Input",
    #     target="Neurons",
    # )

    # input_spikes = (
    #     PoissonEncoder(time=T, dt=parameters.dt)(
    #         0.3 * torch.ones(parameters.batch_size, parameters.features)
    #     )
    #     .to(parameters.device)
    #     .float()
    # )

    # input_data = {"Input": input_spikes}
    # network.to(parameters.device)
    # start = time.time()
    # network.run(inputs=input_data, time=T)
    # end = time.time()

    # duration = end - start
    # return duration
