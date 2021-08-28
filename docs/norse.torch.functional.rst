norse.torch.functional
======================

.. contents:: norse.torch.functional
    :depth: 2
    :local:
    :backlinks: top

    
Temporal operations
-------------------

.. currentmodule:: norse.torch.functional
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    lift <lift.lift>


Threshold functions
-------------------

.. currentmodule:: norse.torch.functional
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    heaviside <heaviside.heaviside>
    heavi_erfc_fn <threshold.heavi_erfc_fn>
    heavi_tanh_fn <threshold.heavi_tanh_fn>
    logistic_fn <threshold.logistic_fn>
    heavi_circ_fn <threshold.heavi_circ_fn>
    circ_dist_fn <threshold.circ_dist_fn>
    triangle_fn <threshold.triangle_fn>
    super_fn <superspike.super_fn>

Neuron models
-------------

Leaky integrator
^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional.leaky_integrator
.. autosummary::
    :toctree: _autosummary
    :nosignatures:
 
    LIParameters
    LIState
    li_feed_forward_step
    

Leaky integrate-and-fire (LIF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional.lif
.. autosummary::
    :toctree: _autosummary
    :nosignatures:
    
    LIFParameters
    LIFState
    lif_feed_forward_integral
    lif_feed_forward_step
    lif_feed_forward_step_sparse
    lif_feed_forward_adjoint_step <norse.torch.functional.adjoint.lif_adjoint.lif_feed_forward_adjoint_step>
    lif_feed_forward_adjoint_step_sparse <norse.torch.functional.adjoint.lif_adjoint.lif_feed_forward_adjoint_step_sparse>



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



Izhikevich
^^^^^^^^^^

.. currentmodule:: norse.torch.module.izhikevich
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    Izhikevich
    IzhikevichCell
    IzhikevichRecurrent
    IzhikevichRecurrentCell



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
^^^^^^^^^^^^^^^^^^^^^^

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
