norse.torch.functional
======================

.. contents:: norse.torch.functional
    :depth: 2
    :local:
    :backlinks: top


Encoding
--------
    
.. currentmodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    constant_current_lif_encode
    gaussian_rbf
    euclidean_distance
    population_encode
    poisson_encode
    poisson_encode_step
    signed_poisson_encode
    signed_poisson_encode_step
    spike_latency_lif_encode
    spike_latency_encode
    lif_current_encoder
    lif_adex_current_encoder
    lif_ex_current_encoder

Logical
-------

.. currentmodule:: norse.torch.functional.logical
.. autosummary::
    :toctree: generated
    :nosignatures:

    logical_and
    logical_xor
    logical_or
    muller_c
    posedge_detector

Regularization
--------------

.. currentmodule:: norse.torch.functional.regularization
.. autosummary::
    :toctree: generated
    :nosignatures:

    regularize_step
    spike_accumulator
    voltage_accumulator


Threshold functions
-------------------

.. currentmodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    heaviside <heaviside.heaviside>
    heavi_erfc_fn <threshold.heavi_erfc_fn>
    heavi_tanh_fn <threshold.heavi_tanh_fn>
    logistic_fn <threshold.logistic_fn>
    heavi_circ_fn <threshold.heavi_circ_fn>
    circ_dist_fn <threshold.circ_dist_fn>
    triangle_fn <threshold.triangle_fn>
    super_fn <superspike.super_fn>

    
Temporal operations
-------------------

.. currentmodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    lift <lift.lift>


Neuron models
-------------

Integrate-and-fire (IAF)
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional.iaf
.. autosummary::
    :toctree: generated
    :nosignatures:

    IAFParameters
    IAFFeedForwardState
    iaf_feed_forward_step

Izhikevich
^^^^^^^^^^

.. currentmodule:: norse.torch.functional.izhikevich
.. autosummary::
    :toctree: generated
    :nosignatures:

    IzhikevichParameters
    IzhikevichSpikingBehavior
    tonic_spiking
    tonic_bursting
    phasic_spiking
    phasic_bursting
    mixed_mode
    spike_frequency_adaptation
    class_1_exc
    class_2_exc
    spike_latency
    subthreshold_oscillation
    resonator
    integrator,
    rebound_spike,
    rebound_burst,
    threshhold_variability,
    bistability,
    dap,
    accomodation,
    inhibition_induced_spiking,
    inhibition_induced_bursting,
    izhikevich_feed_forward_step

Leaky integrator
^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional.leaky_integrator
.. autosummary::
    :toctree: generated
    :nosignatures:
    
    LIParameters
    LIState
    li_feed_forward_step
    

Leaky integrate-and-fire (LIF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:
    
    LIFParameters
    LIFFeedForwardState
    lif_feed_forward_integral
    lif_feed_forward_step
    lif_feed_forward_step_sparse
    lif_feed_forward_adjoint_step
    lif_feed_forward_adjoint_step_sparse

    
LIF, conductance based
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional.coba_lif
.. autosummary::
    :toctree: generated
    :nosignatures:

    CobaLIFParameters
    CobaLIFFeedForwardState
    coba_lif_feed_forward_step
    

LIF, adaptive exponential
^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional.lif_adex
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFAdExParameters
    LIFAdExFeedForwardState

    lif_adex_feed_forward_step
    lif_adex_current_encoder

LIF, exponential
^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional.lif_ex
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFExParameters
    LIFExFeedForwardState
    lif_ex_feed_forward_step
    lif_ex_current_encoder

LIF, multicompartmental (MC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    lif_mc_feed_forward_step
    lif_mc_refrac_feed_forward_step

LIF, refractory
^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFRefracParameters
    LIFRefracFeedForwardState
    lif_refrac_feed_forward_step
    lif_refrac_feed_forward_adjoint_step

Long short-term memory (LSNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    LSNNParameters
    LSNNFeedForwardState
    lsnn_feed_forward_step
    lsnn_feed_forward_adjoint_step


Plasticity models
-----------------

Spike-time dependent plasticity (STDP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    STDPSensorParameters
    STDPSensorState
    stdp_sensor_step

Tsodyks-Markram timing-dependent plasticity (TDP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    TsodyksMakramParameters
    TsodyksMakramState
    stp_step

