"""
Modules for spiking neural network, adhering to the ``torch.nn.Module`` interface.
"""

from .lif_adex_refrac import (
    LIFAdExRefracParameters,
    LIFAdExRefracCell,
    LIFAdExRefracRecurrentCell,
    LIFAdExRefracRecurrent,
    LIFAdExRefracFeedForwardState,
    LIFAdExRefracState,
)
from .coba_lif import CobaLIFCell, CobaLIFParameters, CobaLIFState
from .conv import LConv2d
from .encode import (
    ConstantCurrentLIFEncoder,
    PoissonEncoder,
    PopulationEncoder,
    SignedPoissonEncoder,
    SpikeLatencyEncoder,
    SpikeLatencyLIFEncoder,
)
from .iaf import IAFFeedForwardState, IAFCell, IAFParameters
from .leaky_integrator import LI, LICell, LILinearCell, LIParameters, LIState
from .lif import (
    LIFCell,
    LIFRecurrentCell,
    LIF,
    LIFFeedForwardState,
    LIFParameters,
    LIFState,
    LIFRecurrent,
)
from .lif_adex import (
    LIFAdExCell,
    LIFAdExRecurrentCell,
    LIFAdEx,
    LIFAdExRecurrent,
    LIFAdExFeedForwardState,
    LIFAdExParameters,
    LIFAdExState,
)
from .lif_box import LIFBoxFeedForwardState, LIFBoxParameters, LIFBoxCell
from .lif_correlation import (
    LIFCorrelation,
    LIFCorrelationParameters,
    LIFCorrelationState,
)
from .lif_ex import (
    LIFExCell,
    LIFExRecurrentCell,
    LIFEx,
    LIFExRecurrent,
    LIFExFeedForwardState,
    LIFExParameters,
    LIFExState,
)
from .lif_mc import LIFMCRecurrentCell
from .lif_mc_refrac import LIFMCRefracRecurrentCell
from .lif_refrac import (
    LIFRefracCell,
    LIFRefracRecurrentCell,
    LIFRefracRecurrent,
    LIFRefracFeedForwardState,
    LIFRefracParameters,
    LIFRefracState,
)
from .lift import Lift
from .lsnn import (
    LSNNCell,
    LSNNRecurrentCell,
    LSNN,
    LSNNRecurrent,
    LSNNFeedForwardState,
    LSNNParameters,
    LSNNState,
)
from .regularization import RegularizationCell
from .sequential import SequentialState

from .izhikevich import (
    IzhikevichCell,
    IzhikevichRecurrentCell,
    Izhikevich,
    IzhikevichRecurrent,
)
from .spikes_to_times_decoder import SpikesToTimesDecoder

__all__ = [
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
    "SpikeLatencyEncoder",
    "SpikeLatencyLIFEncoder",
    # Integrate-and-fire
    "IAFFeedForwardState",
    "IAFCell",
    "IAFParameters",
    # Izhikevich
    "IzhikevichCell",
    "IzhikevichRecurrentCell",
    "Izhikevich",
    "IzhikevichRecurrent",
    # Leaky integrator
    "LI",
    "LICell",
    "LILinearCell",
    "LIParameters",
    "LIState",
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
    # Decoder
    "SpikesToTimesDecoder",
]
