norse.torch.functional
======================

.. contents:: norse.torch.functional
    :depth: 2
    :local:
    :backlinks: top


Encoding
--------
    
.. automodule:: norse.torch.functional
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

.. automodule:: norse.torch.functional.logical
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

.. automodule:: norse.torch.functional.regularization
.. autosummary::
    :toctree: generated
    :nosignatures:

    regularize_step
    spike_accumulator
    voltage_accumulator

Threshold functions
-------------------

.. automodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    heaviside
    heavi_erfc_fn
    heavi_tanh_fn
    logistic_fn
    heavi_circ_fn
    circ_dist_fn
    triangle_fn
    super_fn

    
Temporal operations
-------------------

.. automodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    lift <lift.lift>


Neuron models
-------------

Integrate-and-fire (IAF)
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.functional.iaf
.. autosummary::
    :toctree: generated
    :nosignatures:

    IAFParameters
    IAFFeedForwardState
    iaf_feed_forward_step

Izhikevich
^^^^^^^^^^

.. automodule:: norse.torch.functional.izhikevich
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

.. automodule:: norse.torch.functional.leaky_integrator
.. autosummary::
    :toctree: generated
    :nosignatures:
    
    LIParameters
    LIState
    li_feed_forward_step
    

Leaky integrate-and-fire (LIF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.functional
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

LIF, box model
^^^^^^^^^^^^^^

A simplified version of the popular leaky integrate-and-fire neuron model that combines a :mod:`norse.torch.functional.leaky_integrator` with spike thresholds to produce events (spikes).
Compared to the :mod:`norse.torch.functional.lif` modules, this model leaves out the current term, making it computationally simpler but impossible to implement in physical systems because currents cannot "jump" in nature.
It is these sudden current jumps that gives the model its name, because the shift in current is instantaneous and can be drawn as "current boxes".

.. automodule:: norse.torch.functional.lif_box
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFBoxFeedForwardState
    LIFBoxParameters

    lif_box_feed_forward_step

LIF, conductance based
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.functional.coba_lif
.. autosummary::
    :toctree: generated
    :nosignatures:

    CobaLIFParameters
    CobaLIFFeedForwardState
    coba_lif_feed_forward_step
    

LIF, adaptive exponential
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.functional.lif_adex
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFAdExParameters
    LIFAdExFeedForwardState

    lif_adex_feed_forward_step
    lif_adex_current_encoder

LIF, exponential
^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.functional.lif_ex
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFExParameters
    LIFExFeedForwardState
    lif_ex_feed_forward_step
    lif_ex_current_encoder

LIF, multicompartmental (MC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    lif_mc_feed_forward_step
    lif_mc_refrac_feed_forward_step

LIF, refractory
^^^^^^^^^^^^^^^

.. automodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFRefracParameters
    LIFRefracFeedForwardState
    lif_refrac_feed_forward_step
    lif_refrac_feed_forward_adjoint_step

Long short-term memory (LSNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    LSNNParameters
    LSNNFeedForwardState
    lsnn_feed_forward_step
    lsnn_feed_forward_adjoint_step


Receptive fields
----------------

.. automodule:: norse.torch.functional.receptive_field
.. autosummary::
    :toctree: generated

    gaussian_kernel
    spatial_receptive_field
    spatial_receptive_fields_with_derivatives
    temporal_scale_distribution

Plasticity models
-----------------

Spike-time dependent plasticity (STDP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    STDPSensorParameters
    STDPSensorState
    stdp_sensor_step

Tsodyks-Markram timing-dependent plasticity (TDP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: norse.torch.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    TsodyksMakramParameters
    TsodyksMakramState
    stp_step
