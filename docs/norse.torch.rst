norse.torch
===========

Building blocks for spiking neural networks based on `PyTorch <https://pytorch.org>`_.

.. contents:: norse.torch
    :depth: 2
    :local:
    :backlinks: top


Containers
----------

.. currentmodule:: norse.torch.module
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    Lift <lift.Lift>
    SequentialState <sequential.SequentialState>
    RegularizationCell <regularization.RegularizationCell>


Threshold models
----------------

.. currentmodule:: norse.torch.functional
.. autosummary::
    :toctree:
    :nosignatures:

    HeaviErfc <threshold.HeaviErfc>
    HeaviTanh <threshold.HeaviTanh>
    Logistic <threshold.Logistic>
    HeaviCirc <threshold.HeaviCirc>
    CircDist <threshold.CircDist>
    Triangle <threshold.Triangle>
    SuperSpike <superspike.SuperSpike>


Encoding
--------
    
.. currentmodule:: norse.torch.module.encode
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    ConstantCurrentLIFEncoder
    PoissonEncoder
    PoissonEncoderStep
    PopulationEncoder
    SignedPoissonEncoder
    SpikeLatencyEncoder
    SpikeLatencyLIFEncoder


Convolutions
------------

.. currentmodule:: norse.torch.module.conv
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    LConv2d


Neuron models
-------------


Izhikevich
^^^^^^^^^^

.. currentmodule:: norse.torch.module.izhikevich
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    IzhikevichParameters <norse.torch.functional.izhikevich.IzhikevichParameters>
    IzhikevichState <norse.torch.functional.izhikevich.IzhikevichState>

    Izhikevich
    IzhikevichCell
    IzhikevichRecurrent
    IzhikevichRecurrentCell

Leaky integrator
^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.module.leaky_integrator
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    LICell
    LILinearCell


Leaky integrate-and-fire (LIF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    LIFParameters <functional.lif.LIFParameters>
    LIFState <functional.lif.LIFState>

    LIF  <module.lif.LIF>
    LIFCell <module.lif.LIFCell>
    LIFRecurrent <module.lif.LIFRecurrent>
    LIFRecurrentCell <module.lif.LIFRecurrentCell>



LIF, conductance based
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.module.coba_lif
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    CobaLIFCell
    

LIF, adaptive exponential
^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.module.lif_adex
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    LIFAdEx
    LIFAdExCell
    LIFAdExRecurrent
    LIFAdExRecurrentCell

LIF, exponential
^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.module.lif_ex
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    LIFEx
    LIFExCell
    LIFExRecurrent
    LIFExRecurrentCell

Long short-term memory (LSNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.module.lsnn
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    LSNN
    LSNNCell
    LSNNRecurrent
    LSNNRecurrentCell
