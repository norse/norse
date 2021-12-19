"""
Stateless spiking neural network components.
"""

from .adjoint.lif_adjoint import (
    lif_feed_forward_adjoint_step_sparse,
    lif_feed_forward_adjoint_step,
    LIFFeedForwardAdjointFunction,
    LIFFeedForwardSparseAdjointFunction,
)
from .adjoint.lif_refrac_adjoint import (
    lif_refrac_feed_forward_adjoint_step,
    lif_refrac_feed_forward_step,
    LIFAdjointRefracFeedForwardFunction,
)
from .adjoint.lsnn_adjoint import (
    lsnn_feed_forward_adjoint_step,
    LSNNFeedForwardAdjointFunction,
)

from .coba_lif import (
    CobaLIFState,
    CobaLIFParameters,
    CobaLIFFeedForwardState,
    coba_lif_feed_forward_step,
    coba_lif_step,
)
from .correlation_sensor import (
    CorrelationSensorParameters,
    CorrelationSensorState,
    correlation_based_update,
    correlation_sensor_step,
)
from .encode import (
    constant_current_lif_encode,
    euclidean_distance,
    gaussian_rbf,
    poisson_encode,
    poisson_encode_step,
    population_encode,
    signed_poisson_encode,
    signed_poisson_encode_step,
    spike_latency_encode,
    spike_latency_lif_encode,
)
from .heaviside import heaviside

from .iaf import (
    IAFFeedForwardState,
    IAFParameters,
    IAFState,
    iaf_step,
    iaf_feed_forward_step,
)
from .leaky_integrator import LIParameters, LIState, li_feed_forward_step, li_step
from .lif import (
    LIFFeedForwardState,
    LIFParameters,
    LIFState,
    lif_current_encoder,
    lif_feed_forward_step,
    lif_step,
)
from .lif_adex import (
    LIFAdExFeedForwardState,
    LIFAdExParameters,
    LIFAdExState,
    lif_adex_current_encoder,
    lif_adex_feed_forward_step,
    lif_adex_step,
)
from .lif_box import (
    LIFBoxParameters,
    LIFBoxState,
    LIFBoxFeedForwardState,
    lif_box_feed_forward_step,
)
from .lif_correlation import (
    LIFCorrelationParameters,
    LIFCorrelationState,
    lif_correlation_step,
)
from .lif_ex import (
    LIFExFeedForwardState,
    LIFExParameters,
    LIFExState,
    lif_ex_current_encoder,
    lif_ex_feed_forward_step,
    lif_ex_step,
)
from .lif_mc import lif_mc_feed_forward_step, lif_mc_step
from .lif_mc_refrac import lif_mc_refrac_feed_forward_step, lif_mc_refrac_step
from .lif_refrac import (
    LIFRefracFeedForwardState,
    LIFRefracParameters,
    LIFRefracState,
    lif_refrac_feed_forward_step,
    lif_refrac_step,
)
from .lift import lift
from .logical import logical_and, logical_or, logical_xor
from .lsnn import (
    LSNNFeedForwardState,
    LSNNParameters,
    LSNNState,
    lsnn_feed_forward_step,
    lsnn_step,
)
from .regularization import regularize_step, spike_accumulator, voltage_accumulator
from .stdp_sensor import STDPSensorParameters, STDPSensorState, stdp_sensor_step
from .threshold import (
    circ_dist_fn,
    heavi_circ_fn,
    heavi_erfc_fn,
    heavi_tanh_fn,
    triangle_fn,
    logistic_fn,
)

from .izhikevich import (
    IzhikevichParameters,
    IzhikevichSpikingBehavior,
    IzhikevichState,
    IzhikevichRecurrentState,
    create_izhikevich_spiking_behavior,
    tonic_spiking,
    tonic_bursting,
    phasic_spiking,
    phasic_bursting,
    mixed_mode,
    spike_frequency_adaptation,
    class_1_exc,
    class_2_exc,
    spike_latency,
    subthreshold_oscillation,
    resonator,
    integrator,
    rebound_spike,
    rebound_burst,
    threshold_variability,
    bistability,
    dap,
    accomodation,
    inhibition_induced_spiking,
    inhibition_induced_bursting,
    izhikevich_feed_forward_step,
    izhikevich_recurrent_step,
)

from .superspike import super_fn

from .tsodyks_makram import TsodyksMakramParameters, TsodyksMakramState, stp_step

__all__ = [
    # Activation functions
    "heaviside",
    "circ_dist_fn",
    "heavi_circ_fn",
    "heavi_erfc_fn",
    "heavi_tanh_fn",
    "triangle_fn",
    "logistic_fn",
    "super_fn",
    # Adjoint LIF
    "LIFFeedForwardAdjointFunction",
    "LIFFeedForwardSparseAdjointFunction",
    "lif_feed_forward_adjoint_step",
    "lif_feed_forward_adjoint_step_sparse",
    # Adjoint refractory LIF
    "LIFAdjointRefracFeedForwardFunction",
    "lif_refrac_feed_forward_adjoint_step",
    # Adjoint LSNN
    "LSNNFeedForwardAdjointFunction",
    "lsnn_feed_forward_adjoint_step",
    # Correlation sensor
    "CorrelationSensorParameters",
    "CorrelationSensorState",
    "correlation_based_update",
    "correlation_sensor_step",
    # Encoders
    "constant_current_lif_encode",
    "euclidean_distance",
    "gaussian_rbf",
    "poisson_encode",
    "poisson_encode_step",
    "population_encode",
    "signed_poisson_encode",
    "signed_poisson_encode_step",
    "spike_latency_encode",
    "spike_latency_lif_encode",
    # IAF
    "IAFFeedForwardState",
    "IAFParameters",
    "IAFState",
    "iaf_step",
    "iaf_feed_forward_step",
    # Izhikevich
    "IzhikevichParameters",
    "IzhikevichSpikingBehavior",
    "IzhikevichState",
    "IzhikevichRecurrentState",
    "create_izhikevich_spiking_behavior",
    "tonic_spiking",
    "tonic_bursting",
    "phasic_spiking",
    "phasic_bursting",
    "mixed_mode",
    "spike_frequency_adaptation",
    "class_1_exc",
    "class_2_exc",
    "spike_latency",
    "subthreshold_oscillation",
    "resonator",
    "integrator",
    "rebound_spike",
    "rebound_burst",
    "threshold_variability",
    "bistability",
    "dap",
    "accomodation",
    "inhibition_induced_spiking",
    "inhibition_induced_bursting",
    "izhikevich_feed_forward_step",
    "izhikevich_recurrent_step",
    # Leaky integrator
    "LIParameters",
    "LIState",
    "li_feed_forward_step",
    "li_step",
    # LIF
    "LIFFeedForwardState",
    "LIFParameters",
    "LIFState",
    "lif_current_encoder",
    "lif_feed_forward_step",
    "lif_step",
    # LIF Conductance based
    "CobaLIFState",
    "CobaLIFParameters",
    "CobaLIFFeedForwardState",
    "coba_lif_feed_forward_step",
    "coba_lif_step",
    # LIF AdEx
    "LIFAdExFeedForwardState",
    "LIFAdExParameters",
    "LIFAdExState",
    "lif_adex_current_encoder",
    "lif_adex_feed_forward_step",
    "lif_adex_step",
    # LIF box
    "LIFBoxParameters",
    "LIFBoxState",
    "LIFBoxFeedForwardState",
    "lif_box_feed_forward_step",
    # LIF Correlation
    "LIFCorrelationParameters",
    "LIFCorrelationState",
    "lif_correlation_step",
    # LIF Ex
    "LIFExFeedForwardState",
    "LIFExParameters",
    "LIFExState",
    "lif_ex_current_encoder",
    "lif_ex_feed_forward_step",
    "lif_ex_step",
    # LIF MC
    "lif_mc_feed_forward_step",
    "lif_mc_step",
    "lif_mc_refrac_feed_forward_step",
    "lif_mc_refrac_step",
    # Lifting
    "lift",
    # LIF refraec
    "LIFRefracFeedForwardState",
    "LIFRefracParameters",
    "LIFRefracState",
    "lif_refrac_step",
    "lif_refrac_feed_forward_step",
    "lif_refrac_feed_forward_adjoint_step",
    # Logic
    "logical_and",
    "logical_or",
    "logical_xor",
    # LSNN
    "LSNNFeedForwardState",
    "LSNNParameters",
    "LSNNState",
    "lsnn_step",
    "lsnn_feed_forward_step",
    "lsnn_feed_forward_adjoint_step",
    # Regularization
    "regularize_step",
    "spike_accumulator",
    "voltage_accumulator",
    # STDP
    "STDPSensorParameters",
    "STDPSensorState",
    "stdp_sensor_step",
    # TDP
    "TsodyksMakramParameters",
    "TsodyksMakramState",
    "stp_step",
]
