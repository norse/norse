norse.torch
===========

Building blocks for spiking neural networks based on `PyTorch <https://pytorch.org>`_.

.. contents:: norse.torch
    :depth: 2
    :local:
    :backlinks: top


Containers
----------

.. automodule:: norse.torch.module
.. autosummary::
    :toctree: generated
    :nosignatures:

    Lift <lift.Lift>
    SequentialState <sequential.SequentialState>
    RegularizationCell <regularization.RegularizationCell>

Convolutions
------------

.. automodule:: norse.torch.module.conv
.. autosummary::
    :toctree: generated
    :nosignatures:

    LConv2d


.. TODO: After threshold implementation
.. Threshold models
.. ----------------

.. .. automodule:: norse.torch
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
    
.. automodule:: norse.torch.module.encode
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


Neuron models
-------------

Integrate-and-fire
^^^^^^^^^^^^^^^^^^

Simple integrators that sums up incoming signals until a threshold.

.. automodule:: norse.torch.module.iaf
.. autosummary::
    :toctree: generated
    :nosignatures:

    IAFFeedForwardState
    IAFParameters
    IAFCell

Izhikevich
^^^^^^^^^^

.. automodule:: norse.torch.module.izhikevich
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

.. automodule:: norse.torch.module.leaky_integrator
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIState
    LIParameters

    LI
    LICell
    LILinearCell


Leaky integrate-and-fire (LIF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.module.lif
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFParameters <functional.lif.LIFParameters>
    LIFState <functional.lif.LIFState>

    LIF  <module.lif.LIF>
    LIFCell <module.lif.LIFCell>
    LIFRecurrent <module.lif.LIFRecurrent>
    LIFRecurrentCell <module.lif.LIFRecurrentCell>


LIF, box model
^^^^^^^^^^^^^^

.. automodule:: norse.torch.module.lif_box
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFBoxFeedForwardState
    LIFBoxParameters

    LIFBoxCell

LIF, conductance based
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.module.coba_lif
.. autosummary::
    :toctree: generated
    :nosignatures:

    CobaLIFCell
    

LIF, adaptive exponential
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.module.lif_adex
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFAdEx
    LIFAdExCell
    LIFAdExRecurrent
    LIFAdExRecurrentCell

LIF, exponential
^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.module.lif_ex
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFEx
    LIFExCell
    LIFExRecurrent
    LIFExRecurrentCell

LIF, multicompartmental
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.module.lif_mc
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFMCRecurrentCell

LIF, multicompartmental with refraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.module.lif_mc_refrac
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFMCRefracRecurrentCell

LIF, refractory
^^^^^^^^^^^^^^^

.. automodule:: norse.torch.module.lif_refrac
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFRefracCell
    LIFRefracRecurrentCell
    
Long short-term memory (LSNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.module.lsnn
.. autosummary::
    :toctree: generated
    :nosignatures:

    LSNN
    LSNNCell
    LSNNRecurrent
    LSNNRecurrentCell

Receptive fields
----------------

.. automodule:: norse.torch.module.receptive_field
.. autosummary::
    :toctree: generated
    :nosignatures:

    SpatialReceptiveField2d
    TemporalReceptiveField
