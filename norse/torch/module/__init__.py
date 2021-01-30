"""
Modules for spiking neural network, adhering to the ``torch.nn.Module`` interface.
"""

from .coba_lif import CobaLIFCell, CobaLIFParameters, CobaLIFState
from .encode import (
    ConstantCurrentLIFEncoder,
    PoissonEncoder,
    PopulationEncoder,
    SignedPoissonEncoder,
    SpikeLatencyEncoder,
    SpikeLatencyLIFEncoder,
)
from .leaky_integrator import LICell, LIFeedForwardCell, LIParameters, LIState
from .lif import (
    LIFCell,
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
    LIFExFeedForwardCell,
    LIFExFeedForwardState,
    LIFExLayer,
    LIFExParameters,
    LIFExState,
)
from .lif_mc import LIFMCCell
from .lif_mc_refrac import LIFMCRefracCell
from .lif_refrac import (
    LIFRefracCell,
    LIFRefracFeedForwardCell,
    LIFRefracFeedForwardState,
    LIFRefracParameters,
    LIFRefracState,
)
from .lift import Lift
from .lsnn import (
    LSNNCell,
    LSNNFeedForwardCell,
    LSNNFeedForwardState,
    LSNNParameters,
    LSNNState,
    LSNNLayer,
)
from .regularization import RegularizationCell
from .sequential import SequentialState

__all__ = [
    "CobaLIFCell",
    "CobaLIFState",
    "CobaLIFParameters",
    "ConstantCurrentLIFEncoder",
    "PoissonEncoder",
    "PopulationEncoder",
    "SignedPoissonEncoder",
    "SpikeLatencyEncoder",
    "SpikeLatencyLIFEncoder",
    "LICell",
    "LIFeedForwardCell",
    "LIParameters",
    "LIState",
    "LIFCell",
    "LIFRecurrent",
    "LIF",
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
    "LIFExFeedForwardCell",
    "LIFExFeedForwardState",
    "LIFExLayer",
    "LIFExParameters",
    "LIFExState",
    "LIFMCCell",
    "LIFMCRefracCell",
    "LIFRefracParameters",
    "LIFRefracState",
    "LIFMCRefracCell",
    "LIFRefracParameters",
    "LIFRefracState",
    "LIFRefracCell",
    "LIFRefracFeedForwardCell",
    "LIFRefracFeedForwardState",
    "LIFRefracParameters",
    "LIFRefracState",
    "Lift",
    "LSNNCell",
    "LSNNFeedForwardCell",
    "LSNNFeedForwardState",
    "LSNNParameters",
    "LSNNLayer",
    "LSNNState",
    "RegularizationCell",
    "SequentialState",
]
