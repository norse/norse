"""
Modules for spiking neural network, adhering to the ``torch.nn.Module`` interface.
"""

from .coba_lif import CobaLIFCell, CobaLIFParameters, CobaLIFState
from .conv import LConv2d
from .decode import SpikeTimeDecoder
from .encode import (
    ConstantCurrentLIFEncoder,
    PoissonEncoder,
    PopulationEncoder,
    SignedPoissonEncoder,
    SpikeLatencyEncoder,
    SpikeLatencyLIFEncoder,
)
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

__all__ = [
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
    "SpikeTimeDecoder",
    "LI",
    "LICell",
    "LILinearCell",
    "LIParameters",
    "LIState",
    "LIF",
    "LIFCell",
    "LIFRecurrent",
    "LIFRecurrentCell",
    "LIFFeedForwardState",
    "LIFParameters",
    "LIFState",
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
    "LIFCorrelation",
    "LIFCorrelationParameters",
    "LIFCorrelationState",
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
    "IzhikevichCell",
    "IzhikevichRecurrentCell",
    "Izhikevich",
    "IzhikevichRecurrent",
]
