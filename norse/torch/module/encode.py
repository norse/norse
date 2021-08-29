"""
Stateful encoders as torch modules.
"""

from typing import Union, Callable
import torch

from norse.torch.functional import lif
from norse.torch.functional import encode


class ConstantCurrentLIFEncoder(torch.nn.Module):
    """Encodes input currents as fixed (constant) voltage currents, and simulates the spikes that
    occur during a number of timesteps/iterations (seq_length).

    Example:
    >>> data = torch.as_tensor([2, 4, 8, 16])
    >>> seq_length = 2 # Simulate two iterations
    >>> constant_current_lif_encode(data, seq_length)
    (tensor([[0.2000, 0.4000, 0.8000, 0.0000],   # State in terms of membrane voltage
                [0.3800, 0.7600, 0.0000, 0.0000]]),
    tensor([[0., 0., 0., 1.],                   # Spikes for each iteration
            [0., 0., 1., 1.]]))

    Parameters:
        seq_length (int): The number of iterations to simulate
        p (LIFParameters): Initial neuron parameters.
        dt (float): Time delta between simulation steps
    """

    def __init__(
        self,
        seq_length: int,
        p: lif.LIFParameters = lif.LIFParameters(),
        dt: float = 0.001,
    ):
        super(ConstantCurrentLIFEncoder, self).__init__()
        self.seq_length = seq_length
        self.p = p
        self.dt = dt

    def forward(self, input_currents):
        return encode.constant_current_lif_encode(
            input_currents,
            seq_length=self.seq_length,
            p=self.p,
            dt=self.dt,
        )


class PoissonEncoder(torch.nn.Module):
    """
    Encodes a tensor of input values, which are assumed to be in the
    range [0,1] into a tensor of one dimension higher of binary values,
    which represent input spikes.

    Parameters:
        sequence_length (int): Number of time steps in the resulting spike train.
        f_max (float): Maximal frequency (in Hertz) which will be emitted.
        dt (float): Integration time step (should coincide with the integration time step used in the model)
    """

    def __init__(self, seq_length: int, f_max: float = 100, dt: float = 0.001):
        super(PoissonEncoder, self).__init__()
        self.seq_length = seq_length
        self.f_max = f_max
        self.dt = dt

    def forward(self, x):
        return encode.poisson_encode(x, self.seq_length, f_max=self.f_max, dt=self.dt)


class PoissonEncoderStep(torch.nn.Module):
    """Encodes a tensor of input values, which are assumed to be in the
    range [0,1] into a tensor of binary values, which represent input spikes.

    Parameters:
        f_max (float): Maximal frequency (in Hertz) which will be emitted.
        dt (float): Integration time step (should coincide with the integration time step used in the model)
    """

    def __init__(self, f_max: float = 1000, dt: float = 0.001):
        super(PoissonEncoderStep, self).__init__()
        self.f_max = f_max
        self.dt = dt

    def forward(self, x):
        return encode.poisson_encode_step(x, f_max=self.f_max, dt=self.dt)


class PopulationEncoder(torch.nn.Module):
    """Encodes a set of input values into population codes, such that each singular input value is represented by
    a list of numbers (typically calculated by a radial basis kernel), whose length is equal to the out_features.

    Population encoding can be visualised by imagining a number of neurons in a list, whose activity increases
    if a number gets close to its "receptive field".

    .. figure:: https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/PopulationCode.svg/1920px-PopulationCode.svg.png

        Gaussian curves representing different neuron "receptive fields". Image credit: `Andrew K. Richardson`_.

    .. _Andrew K. Richardson: https://commons.wikimedia.org/wiki/File:PopulationCode.svg

    Example:
        >>> data = torch.as_tensor([0, 0.5, 1])
        >>> out_features = 3
        >>> PopulationEncoder(out_features).forward(data)
        tensor([[1.0000, 0.8825, 0.6065],
                [0.8825, 1.0000, 0.8825],
                [0.6065, 0.8825, 1.0000]])

    Parameters:
        out_features (int): The number of output *per* input value
        scale (torch.Tensor): The scaling factor for the kernels. Defaults to the maximum value of the input.
                            Can also be set for each individual sample.
        kernel: A function that takes two inputs and returns a tensor. The two inputs represent the center value
                (which changes for each index in the output tensor) and the actual data value to encode respectively.z
                Defaults to gaussian radial basis kernel function.
        distance_function: A function that calculates the distance between two numbers. Defaults to euclidean.
    """

    def __init__(
        self,
        out_features: int,
        scale: Union[int, torch.Tensor] = None,
        kernel: Callable[[torch.Tensor], torch.Tensor] = encode.gaussian_rbf,
        distance_function: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = encode.euclidean_distance,
    ):
        super(PopulationEncoder, self).__init__()
        self.out_features = out_features
        self.scale = scale
        self.kernel = kernel
        self.distance_function = distance_function

    def forward(self, input_tensor):
        return encode.population_encode(
            input_tensor,
            self.out_features,
            self.scale,
            self.kernel,
            self.distance_function,
        )


class SignedPoissonEncoder(torch.nn.Module):
    """Encodes a tensor of input values, which are assumed to be in the
    range [-1,1] into a tensor of one dimension higher of values in {-1,0,1},
    which represent signed input spikes.

    Parameters:
        sequence_length (int): Number of time steps in the resulting spike train.
        f_max (float): Maximal frequency (in Hertz) which will be emitted.
        dt (float): Integration time step (should coincide with the integration time step used in the model)
    """

    def __init__(self, seq_length: int, f_max: float = 100, dt: float = 0.001):
        super(SignedPoissonEncoder, self).__init__()
        self.seq_length = seq_length
        self.f_max = f_max
        self.dt = dt

    def forward(self, x):
        return encode.signed_poisson_encode(
            x, self.seq_length, f_max=self.f_max, dt=self.dt
        )


class SignedPoissonEncoderStep(torch.nn.Module):
    """Encodes a tensor of input values, which are assumed to be in the
    range [-1,1] into a tensor of values in {-1,0,1},
    which represent signed input spikes.

    Parameters:
        f_max (float): Maximal frequency (in Hertz) which will be emitted.
        dt (float): Integration time step (should coincide with the integration time step used in the model)
    """

    def __init__(self, f_max: float = 1000, dt: float = 0.001):
        super(SignedPoissonEncoderStep, self).__init__()
        self.f_max = f_max
        self.dt = dt

    def forward(self, x):
        return encode.signed_poisson_encode_step(x, f_max=self.f_max, dt=self.dt)


class SpikeLatencyLIFEncoder(torch.nn.Module):
    """Encodes an input value by the time the first spike occurs.
    Similar to the ConstantCurrentLIFEncoder, but the LIF can be
    thought to have an infinite refractory period.

    Parameters:
        sequence_length (int): Number of time steps in the resulting spike train.
        p (LIFParameters): Parameters of the LIF neuron model.
        dt (float): Integration time step (should coincide with the integration time step used in the model)
    """

    def __init__(self, seq_length, p=lif.LIFParameters(), dt=0.001):
        super(SpikeLatencyLIFEncoder, self).__init__()
        self.seq_length = seq_length
        self.p = p
        self.dt = dt

    def forward(self, input_current):
        return encode.spike_latency_lif_encode(
            input_current, self.seq_length, self.p, self.dt
        )


class SpikeLatencyEncoder(torch.nn.Module):
    """For all neurons, remove all but the first spike. This encoding basically measures the time it takes for a
    neuron to spike *first*. Assuming that the inputs are constant, this makes sense in that strong inputs spikes
    fast.

    See `R. Van Rullen & S. J. Thorpe (2001): Rate Coding Versus Temporal Order Coding: What the Retinal Ganglion Cells Tell the Visual Cortex <https://doi.org/10.1162/08997660152002852>`_.

    Spikes are identified by their unique position in the input array.

    Example:
    >>> data = torch.as_tensor([[0, 1, 1], [1, 1, 1]])
    >>> encoder = torch.nn.Sequential(
                    ConstantCurrentLIFEncoder()
                    SpikeLatencyEncoder()
                    )
    >>> encoder(data)
    tensor([[0, 1, 1],
            [1, 0, 0]])
    """

    def forward(self, input_spikes):
        return encode.spike_latency_encode(input_spikes)
