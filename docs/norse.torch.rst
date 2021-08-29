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
    :toctree: generated
    :nosignatures:

    Lift <lift.Lift>
    SequentialState <sequential.SequentialState>
    RegularizationCell <regularization.RegularizationCell>


.. TODO: After threshold implementation
.. Threshold models
.. ----------------

.. .. currentmodule:: norse.torch
.. .. autosummary::
..     :toctree: generated
..     :nosignatures:

..     functional.threshold.HeaviErfc
..     functional.threshold.HeaviTanh
..     functional.threshold.Logistic
..     functional.threshold.HeaviCirc
..     functional.threshold.CircDist
..     functional.threshold.Triangle
..     functional.superspike.SuperSpike


Encoding
--------
    
.. currentmodule:: norse.torch.module.encode
.. autosummary::
    :toctree: generated
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
    :toctree: generated
    :nosignatures:

    LConv2d


Neuron models
-------------


Izhikevich
^^^^^^^^^^

.. currentmodule:: norse.torch.module.izhikevich
.. autosummary::
    :toctree: generated
    :nosignatures:

    IzhikevichParameters
    IzhikevichState
    IzhikevichSpikingBehavior

    Izhikevich
    IzhikevichCell
    IzhikevichRecurrent
    IzhikevichRecurrentCell

Leaky integrator
^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.module.leaky_integrator
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIState
    LIParameters

    LICell
    LILinearCell


Leaky integrate-and-fire (LIF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch
.. autosummary::
    :toctree: generated
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
    :toctree: generated
    :nosignatures:

    CobaLIFCell
    

LIF, adaptive exponential
^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.module.lif_adex
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFAdEx
    LIFAdExCell
    LIFAdExRecurrent
    LIFAdExRecurrentCell

LIF, exponential
^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.module.lif_ex
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFEx
    LIFExCell
    LIFExRecurrent
    LIFExRecurrentCell

Long short-term memory (LSNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.module.lsnn
.. autosummary::
    :toctree: generated
    :nosignatures:

    LSNN
    LSNNCell
    LSNNRecurrent
    LSNNRecurrentCell
