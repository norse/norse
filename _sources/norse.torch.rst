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

Integrate-and-fire
^^^^^^^^^^^^^^^^^^

Simple integrators that sums up incoming signals until a threshold.

.. currentmodule:: norse.torch.module.iaf
.. autosummary::
    :toctree: generated
    :nosignatures:

    IAFFeedForwardState
    IAFParameters

    IAFCell

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

.. currentmodule:: norse.torch.module.leaky_integrator
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


LIF, box model
^^^^^^^^^^^^^^

A simplified version of the popular leaky integrate-and-fire neuron model that combines a :mod:`norse.torch.functional.leaky_integrator` with spike thresholds to produce events (spikes).
Compared to the :mod:`norse.torch.functional.lif` modules, this model leaves out the current term, making it computationally simpler but impossible to implement in physical systems because currents cannot "jump" in nature.
It is these sudden current jumps that gives the model its name, because the shift in current is instantaneous and can be drawn as "current boxes".

.. currentmodule:: norse.torch.module.lif_box
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFBoxFeedForwardState
    LIFBoxParameters

    LIFBoxCell

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

LIF, multicompartmental
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.module.lif_mc
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFMCRecurrentCell

LIF, multicompartmental with refraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.module.lif_mc_refrac
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFMCRefracRecurrentCell

LIF, refractory
^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.module.lif_refrac
.. autosummary::
    :toctree: generated
    :nosignatures:

    LIFRefracCell
    LIFRefracRecurrentCell
    
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
