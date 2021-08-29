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
    :toctree: generated
    :nosignatures:

    lift <lift.lift>


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


Encoding
--------
    
.. currentmodule:: norse.torch.functional.encode
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

.. currentmodule:: norse.torch.module.izhikevich
.. autosummary::
    :toctree: generated
    :nosignatures:

    Izhikevich
    IzhikevichCell
    IzhikevichRecurrent
    IzhikevichRecurrentCell


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

.. currentmodule:: norse.torch.functional.lif
.. autosummary::
    :toctree: generated
    :nosignatures:
    
    LIFParameters
    LIFFeedForwardState
    lif_feed_forward_integral
    lif_feed_forward_step
    lif_feed_forward_step_sparse
    lif_feed_forward_adjoint_step <norse.torch.functional.adjoint.lif_adjoint.lif_feed_forward_adjoint_step>
    lif_feed_forward_adjoint_step_sparse <norse.torch.functional.adjoint.lif_adjoint.lif_feed_forward_adjoint_step_sparse>

    
LIF, conductance based
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: norse.torch.functional.coba_lif
.. autosummary::
    :toctree: generated
    :nosignatures:

    CobaLIFFeedForwardState
    coba_lif_feed_forward_step
    

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
^^^^^^^^^^^^^^^^^^^^^^

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
