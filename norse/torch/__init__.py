"""Norse is a library for doing deep learning with spiking neural networks.

This package contains modules that extends PyTorch with spiking neural
network functionality.
"""

################################################
### FUNCTIONAL
################################################
from norse.torch.functional.adjoint.lif_adjoint import (
    lif_feed_forward_adjoint_step_sparse,
    lif_feed_forward_adjoint_step,
    LIFFeedForwardAdjointFunction,
    LIFFeedForwardSparseAdjointFunction,
)
from norse.torch.functional.adjoint.lif_refrac_adjoint import (
    lif_refrac_feed_forward_adjoint_step,
    LIFAdjointRefracFeedForwardFunction,
)
from norse.torch.functional.adjoint.lsnn_adjoint import (
    lsnn_feed_forward_adjoint_step,
    LSNNFeedForwardAdjointFunction,
)

from norse.torch.functional.coba_lif import (
    CobaLIFState,
    CobaLIFParameters,
    CobaLIFFeedForwardState,
    coba_lif_feed_forward_step,
    coba_lif_step,
)
from norse.torch.functional.correlation_sensor import (
    CorrelationSensorParameters,
    CorrelationSensorState,
    correlation_based_update,
    correlation_sensor_step,
)
from norse.torch.functional.encode import (
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
from norse.torch.functional.heaviside import heaviside

from norse.torch.functional.iaf import (
    IAFFeedForwardState,
    IAFParameters,
    IAFState,
    iaf_step,
    iaf_feed_forward_step,
)
from norse.torch.functional.leaky_integrator import (
    LIParameters,
    LIState,
    li_feed_forward_step,
    li_step,
)
from norse.torch.functional.leaky_integrator_box import (
    LIBoxParameters,
    LIBoxState,
    li_box_feed_forward_step,
    li_box_step,
)
from norse.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParameters,
    LIFState,
    lif_current_encoder,
    lif_feed_forward_step,
    lif_step,
)
from norse.torch.functional.lif_adex import (
    LIFAdExFeedForwardState,
    LIFAdExParameters,
    LIFAdExState,
    lif_adex_current_encoder,
    lif_adex_feed_forward_step,
    lif_adex_step,
)
from norse.torch.functional.lif_box import (
    LIFBoxParameters,
    LIFBoxState,
    LIFBoxFeedForwardState,
    lif_box_feed_forward_step,
)
from norse.torch.functional.lif_correlation import (
    LIFCorrelationParameters,
    LIFCorrelationState,
    lif_correlation_step,
)
from norse.torch.functional.lif_ex import (
    LIFExFeedForwardState,
    LIFExParameters,
    LIFExState,
    lif_ex_current_encoder,
    lif_ex_feed_forward_step,
    lif_ex_step,
)
from norse.torch.functional.lif_mc import lif_mc_feed_forward_step, lif_mc_step
from norse.torch.functional.lif_mc_refrac import (
    lif_mc_refrac_feed_forward_step,
    lif_mc_refrac_step,
)
from norse.torch.functional.lif_refrac import (
    LIFRefracFeedForwardState,
    LIFRefracParameters,
    LIFRefracState,
    lif_refrac_feed_forward_step,
    lif_refrac_step,
)
from norse.torch.functional.lift import lift
from norse.torch.functional.logical import logical_and, logical_or, logical_xor
from norse.torch.functional.lsnn import (
    LSNNFeedForwardState,
    LSNNParameters,
    LSNNState,
    lsnn_feed_forward_step,
    lsnn_step,
)
from norse.torch.functional.regularization import (
    regularize_step,
    spike_accumulator,
    voltage_accumulator,
)
from norse.torch.functional.stdp_sensor import (
    STDPSensorParameters,
    STDPSensorState,
    stdp_sensor_step,
)
from norse.torch.functional.threshold import (
    circ_dist_fn,
    heavi_circ_fn,
    heavi_erfc_fn,
    heavi_tanh_fn,
    triangle_fn,
    logistic_fn,
)

from norse.torch.functional.izhikevich import (
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

from norse.torch.functional.receptive_field import (
    gaussian_kernel,
    spatial_receptive_field,
    spatial_receptive_fields_with_derivatives,
)
from norse.torch.functional.reset import ResetMethod, reset_value, reset_subtract

from norse.torch.functional.superspike import super_fn

from norse.torch.functional.tsodyks_makram import (
    TsodyksMakramParameters,
    TsodyksMakramState,
    stp_step,
)


################################################
### MODELS
################################################
from norse.torch.models.conv import ConvNet, ConvNet4
from norse.torch.models.vgg import (
    VGG,
    vgg11,
    vgg11_bn,
    vgg13,
    vgg13_bn,
    vgg16,
    vgg16_bn,
    vgg19,
    vgg19_bn,
)
from norse.torch.models.mobilenet import MobileNetV2, mobilenet_v2


################################################
### MODULES
################################################
from norse.torch.module.lif_adex_refrac import (
    LIFAdExRefracParameters,
    LIFAdExRefracCell,
    LIFAdExRefracRecurrentCell,
    LIFAdExRefracRecurrent,
    LIFAdExRefracFeedForwardState,
    LIFAdExRefracState,
)
from norse.torch.module.coba_lif import CobaLIFCell, CobaLIFParameters, CobaLIFState
from norse.torch.module.conv import LConv2d
from norse.torch.module.encode import (
    ConstantCurrentLIFEncoder,
    PoissonEncoder,
    PopulationEncoder,
    SignedPoissonEncoder,
    SpikeLatencyEncoder,
    SpikeLatencyLIFEncoder,
)
from norse.torch.module.iaf import IAFFeedForwardState, IAF, IAFCell, IAFParameters
from norse.torch.module.leaky_integrator import (
    LI,
    LICell,
    LILinearCell,
    LIParameters,
    LIState,
)
from norse.torch.module.leaky_integrator_box import (
    LIBoxCell,
    LIBoxParameters,
    LIBoxState,
)
from norse.torch.module.lif import (
    LIFCell,
    LIFRecurrentCell,
    LIF,
    LIFFeedForwardState,
    LIFParameters,
    LIFState,
    LIFRecurrent,
)
from norse.torch.module.lif_adex import (
    LIFAdExCell,
    LIFAdExRecurrentCell,
    LIFAdEx,
    LIFAdExRecurrent,
    LIFAdExFeedForwardState,
    LIFAdExParameters,
    LIFAdExState,
)
from norse.torch.module.lif_box import (
    LIFBoxFeedForwardState,
    LIFBoxParameters,
    LIFBoxCell,
)
from norse.torch.module.lif_correlation import (
    LIFCorrelation,
    LIFCorrelationParameters,
    LIFCorrelationState,
)
from norse.torch.module.lif_ex import (
    LIFExCell,
    LIFExRecurrentCell,
    LIFEx,
    LIFExRecurrent,
    LIFExFeedForwardState,
    LIFExParameters,
    LIFExState,
)
from norse.torch.module.lif_mc import LIFMCRecurrentCell
from norse.torch.module.lif_mc_refrac import LIFMCRefracRecurrentCell
from norse.torch.module.lif_refrac import (
    LIFRefracCell,
    LIFRefracRecurrentCell,
    LIFRefracRecurrent,
    LIFRefracFeedForwardState,
    LIFRefracParameters,
    LIFRefracState,
)
from norse.torch.module.lift import Lift
from norse.torch.module.lsnn import (
    LSNNCell,
    LSNNRecurrentCell,
    LSNN,
    LSNNRecurrent,
    LSNNFeedForwardState,
    LSNNParameters,
    LSNNState,
)
from norse.torch.module.regularization import RegularizationCell
from norse.torch.module.sequential import (
    SequentialState,
    RecurrentSequential,
    RecurrentSequentialState,
)

from norse.torch.module.izhikevich import (
    IzhikevichCell,
    IzhikevichRecurrentCell,
    Izhikevich,
    IzhikevichRecurrent,
)
from norse.torch.module.receptive_field import (
    ParameterizedSpatialReceptiveField2d,
    SampledSpatialReceptiveField2d,
    SpatialReceptiveField2d,
    TemporalReceptiveField,
)
from norse.torch.module.spikes_to_times_decoder import SpikesToTimesDecoder

################################################
### UTILS
################################################
import importlib, logging

UTIL_MODULES = []
tensorboard_loader = importlib.util.find_spec("tensorboard")
HAS_TENSORBOARD = tensorboard_loader is not None
if HAS_TENSORBOARD:
    from norse.torch.utils.tensorboard.tensorboard import (
        hook_spike_activity_mean,
        hook_spike_activity_sum,
        hook_spike_histogram_mean,
        hook_spike_histogram_sum,
        hook_spike_image,
    )

    UTIL_MODULES += [
        "hook_spike_activity_mean",
        "hook_spike_activity_sum",
        "hook_spike_histogram_mean",
        "hook_spike_histogram_sum",
        "hook_spike_image",
    ]
else:
    logging.debug(
        f"Failed to import Norse tensorboard utilities: Tensorboard not installed"
    )

matplotlib_loader = importlib.util.find_spec("matplotlib")
HAS_MATPLOTLIB = matplotlib_loader is not None
if HAS_MATPLOTLIB:
    from norse.torch.utils.plot.plot import (
        plot_heatmap_2d,
        plot_heatmap_3d,
        plot_histogram_2d,
        plot_neuron_states,
        plot_scatter_3d,
        plot_spikes_2d,
        plot_izhikevich,
    )

    UTIL_MODULES += [
        "plot_heatmap_2d",
        "plot_heatmap_3d",
        "plot_histogram_2d",
        "plot_neuron_states",
        "plot_scatter_3d",
        "plot_spikes_2d",
        "plot_izhikevich",
    ]
else:
    logging.debug(
        f"Failed to import Norse plotting utilities: Matplotlib not installed"
    )
del logging, importlib, tensorboard_loader, matplotlib_loader

from norse.torch.utils.import_nir import from_nir
from norse.torch.utils.export_nir import to_nir

__all__ = [
    ############ FUNCTIONAL
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
    # Leaky integrator box
    "LIBoxParameters",
    "LIBoxState",
    "li_box_feed_forward_step",
    "li_box_step",
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
    # Receptive fields
    "gaussian_kernel",
    "spatial_receptive_field",
    "spatial_receptive_fields_with_derivatives",
    # Regularization
    "regularize_step",
    "spike_accumulator",
    "voltage_accumulator",
    # Reset
    "ResetMethod",
    "reset_value",
    "reset_subtract",
    # STDP
    "STDPSensorParameters",
    "STDPSensorState",
    "stdp_sensor_step",
    # TDP
    "TsodyksMakramParameters",
    "TsodyksMakramState",
    "stp_step",
    ############ MODELS
    "ConvNet",
    "ConvNet4",
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19",
    "vgg19_bn",
    "MobileNetV2",
    "mobilenet_v2",
    ############ MODULES
    "LIFAdExRefracState",
    "LIFAdExRefracParameters",
    "LIFAdExRefracCell",
    "LIFAdExRefracRecurrentCell",
    "LIFAdExRefracRecurrent",
    "LIFAdExRefracFeedForwardState",
    "CobaLIFCell",
    "CobaLIFState",
    "CobaLIFParameters",
    "ConstantCurrentLIFEncoder",
    "LConv2d",
    "PoissonEncoder",
    "PopulationEncoder",
    "SignedPoissonEncoder",
    "SpatialReceptiveField2d",
    "SpikeLatencyEncoder",
    "SpikeLatencyLIFEncoder",
    # Integrate-and-fire
    "IAFFeedForwardState",
    "IAF",
    "IAFCell",
    "IAFParameters",
    # Izhikevich
    "IzhikevichCell",
    "IzhikevichRecurrentCell",
    "Izhikevich",
    "IzhikevichRecurrent",
    # Leaky integrator
    "LI",
    "LIBoxCell",
    "LICell",
    "LILinearCell",
    "LIParameters",
    "LIState",
    # Leaky integrator box
    "LIBoxCell",
    "LIBoxParameters",
    "LIBoxState",
    # LIF
    "LIF",
    "LIFCell",
    "LIFRecurrent",
    "LIFRecurrentCell",
    "LIFFeedForwardState",
    "LIFParameters",
    "LIFState",
    # LIF AdEx
    "LIFAdEx",
    "LIFAdExCell",
    "LIFAdExFeedForwardState",
    "LIFAdExParameters",
    "LIFAdExState",
    "LIFAdExCell",
    "LIFAdExRecurrentCell",
    "LIFAdExRecurrent",
    "LIFAdExParameters",
    "LIFAdExState",
    # LIF Box
    "LIFBoxFeedForwardState",
    "LIFBoxParameters",
    "LIFBoxCell",
    # LIF Correlation
    "LIFCorrelation",
    "LIFCorrelationParameters",
    "LIFCorrelationState",
    # LIF Ex
    "LIFExCell",
    "LIFExRecurrentCell",
    "LIFEx",
    "LIFExRecurrent",
    "LIFExFeedForwardState",
    "LIFExParameters",
    "LIFExState",
    "LIFMCRecurrentCell",
    "LIFMCRefracRecurrentCell",
    "LIFRefracCell",
    "LIFRefracRecurrentCell",
    "LIFRefracRecurrent",
    "LIFRefracFeedForwardState",
    "LIFRefracParameters",
    "LIFRefracState",
    "Lift",
    "LSNNCell",
    "LSNNRecurrentCell",
    "LSNN",
    "LSNNRecurrent",
    "LSNNFeedForwardState",
    "LSNNParameters",
    "LSNNState",
    "RegularizationCell",
    "SequentialState",
    "RecurrentSequential",
    "RecurrentSequentialState",
    # Receptive fields
    "ParameterizedSpatialReceptiveField2d",
    "SampledSpatialReceptiveField2d",
    "SpatialReceptiveField2d",
    "TemporalReceptiveField",
    # Decoder
    "SpikesToTimesDecoder",
    ############ UTILS
    "from_nir",
    "to_nir",
] + UTIL_MODULES
del UTIL_MODULES
