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

Leaky integrators describe a *leaky* neuron membrane that integrates
incoming currents over time, but never spikes. In other words, the
neuron adds up incoming input current, while leaking out some of it
in every timestep.

.. math::
    \begin{align*}
        \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
        \dot{i} &= -1/\tau_{\text{syn}} i
    \end{align*}

The first equation describes how the membrane voltage (:math:`v`, across
the membrane) changes over time. A constant amount of current is *leaked*
out every timestep (:math:`v_{\text{leak}}`), while the current
(:math:`i`) is added.

The second equation describes how the current flowing into the neuron
changes in every timestep.

Notice that both equations are parameterized by the *time constant*
:math:`\tau`. This constant controls how *fast* the changes in voltage
and current occurs. A large time constant means a small change.
In Norse, we call this parameter the *inverse* to avoid having to
recalculate the inverse (:math:`\tau_{\text{mem_inv}}` and
:math:`\tau_{\text{syn_inv}}` respectively).
So, for Norse a large inverse time constant means *rapid* changes while
a small inverse time constant means *slow* changes.

Recall that *voltage* is the difference in charge between two points (in
this case the neuron membrane) and *current* is the rate of change or the
amount of current being added/subtracted at each timestep.

More information can be found on
`Wikipedia <https://en.wikipedia.org/wiki/Leaky_integrator>`_.

.. currentmodule:: norse.torch.functional.leaky_integrator
.. autosummary::
    :toctree: generated
    :nosignatures:
    
    LIParameters
    LIState
    li_feed_forward_step
    

Leaky integrate-and-fire (LIF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A popular neuron model that combines a :mod:`norse.torch.functional.leaky_integrator` with
spike thresholds to produce events (spikes).

The model describes the change in a neuron membrane voltage (:math:`v`)
and inflow current (:math:`i`).
See the :mod:`.leaky_integrator` module for more information.

.. math::
    \begin{align*}
        \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
        \dot{i} &= 1/\tau_{\text{syn}} i
    \end{align*}

The F in LIF stands for the thresholded "firing" events that occur if the
neuron voltage increases over a certain point or *threshold* (:math:`v_{\text{th}}`).

.. math::
    z = \Theta(v - v_{\text{th}})

In regular artificial neural networks, this is referred to as the *activation
function*. The behaviour can be controlled by setting the :code:`method` field in
the neuron parameters, but will default to the :mod:`.superspike` synthetic
gradient approach that uses the :mod:`.heaviside` step function:

.. math::
    H[n]=\begin{cases} 0, & n <= 0 \\ 1, & n \gt 0 \end{cases}


More information can be found on
`Wikipedia <https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire>`_
or in the book `*Neuron Dynamics* by W. Gerstner et al.,
freely available online <https://neuronaldynamics.epfl.ch/online/Ch5.html>`_.

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

LIF, box model
^^^^^^^^^^^^^^

A simplified version of the popular leaky integrate-and-fire neuron model that combines a :mod:`norse.torch.functional.leaky_integrator` with spike thresholds to produce events (spikes).
Compared to the :mod:`norse.torch.functional.lif` modules, this model leaves out the current term, making it computationally simpler but impossible to implement in physical systems because currents cannot "jump" in nature.
It is these sudden current jumps that gives the model its name, because the shift in current is instantaneous and can be drawn as "current boxes".

.. currentmodule:: norse.torch.functional.lif_box
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFBoxFeedForwardState
    LIFBoxParameters

    lif_box_feed_forward_step

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

