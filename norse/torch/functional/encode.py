"""
Stateless encoding functionality for Norse, offering different ways to convert numerical
inputs to the spiking domain. Note that some functions, like `population_encode` does not return spikes,
but rather numerical values that will have to be converted into spikes via, for instance, the poisson encoder.
"""

from typing import Callable, Union

import torch

from norse.torch.functional.lif import lif_current_encoder, LIFParameters


def constant_current_lif_encode(
    input_current: torch.Tensor,
    seq_length: int,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> torch.Tensor:
    """
    Encodes input currents as fixed (constant) voltage currents, and simulates the spikes that
    occur during a number of timesteps/iterations (seq_length).

    Example:
        >>> data = torch.as_tensor([2, 4, 8, 16])
        >>> seq_length = 2 # Simulate two iterations
        >>> constant_current_lif_encode(data, seq_length)
         # State in terms of membrane voltage
        (tensor([[0.2000, 0.4000, 0.8000, 0.0000],
                 [0.3800, 0.7600, 0.0000, 0.0000]]),
         # Spikes for each iteration
         tensor([[0., 0., 0., 1.],
                 [0., 0., 1., 1.]]))

    Parameters:
        input_current (torch.Tensor): The input tensor, representing LIF current
        seq_length (int): The number of iterations to simulate
        p (LIFParameters): Initial neuron parameters.
        dt (float): Time delta between simulation steps

    Returns:
        A tensor with an extra dimension of size `seq_length` containing spikes (1) or no spikes (0).
    """
    v = torch.zeros(*input_current.shape, device=input_current.device)
    z = torch.zeros(*input_current.shape, device=input_current.device)
    spikes = torch.zeros(seq_length, *input_current.shape, device=input_current.device)

    for ts in range(seq_length):
        z, v = lif_current_encoder(input_current=input_current, voltage=v, p=p, dt=dt)
        spikes[ts] = z
    return spikes


def gaussian_rbf(tensor: torch.Tensor, sigma: float = 1):
    """
    A `gaussian radial basis kernel <https://en.wikipedia.org/wiki/Radial_basis_function_kernel>`_
    that calculates the radial basis given a distance value (distance between :math:`x` and a data
    value :math:`x'`, or :math:`\\|\\mathbf{x} - \\mathbf{x'}\\|^2` below).

    .. math::
        K(\\mathbf{x}, \\mathbf{x'}) = \\exp\\left(- \\frac{\\|\\mathbf{x} - \\mathbf{x'}\\|^2}{2\\sigma^2}\\right)

    Parameters:
        tensor (torch.Tensor): The tensor containing distance values to convert to radial bases
        sigma (float): The spread of the gaussian distribution. Defaults to 1.
    """
    return torch.exp(-tensor / (2 * sigma**2))


def euclidean_distance(x, y):
    """
    Simple euclidean distance metric.
    """
    return (x - y).pow(2)


def population_encode(
    input_values: torch.Tensor,
    out_features: int,
    scale: Union[int, torch.Tensor] = None,
    kernel: Callable[[torch.Tensor], torch.Tensor] = gaussian_rbf,
    distance_function: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ] = euclidean_distance,
) -> torch.Tensor:
    """
    Encodes a set of input values into population codes, such that each singular input value is represented by
    a list of numbers (typically calculated by a radial basis kernel), whose length is equal to the out_features.

    Population encoding can be visualised by imagining a number of neurons in a list, whose activity increases
    if a number gets close to its "receptive field".

    .. figure:: https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/PopulationCode.svg/1920px-PopulationCode.svg.png

        Gaussian curves representing different neuron "receptive fields". Image credit: `Andrew K. Richardson`_.

    .. _Andrew K. Richardson: https://commons.wikimedia.org/wiki/File:PopulationCode.svg

    Example:
        >>> data = torch.as_tensor([0, 0.5, 1])
        >>> out_features = 3
        >>> pop_encoded = population_encode(data, out_features)
        tensor([[1.0000, 0.8825, 0.6065],
                [0.8825, 1.0000, 0.8825],
                [0.6065, 0.8825, 1.0000]])
        >>> spikes = poisson_encode(pop_encoded, 1).squeeze() # Convert to spikes

    Parameters:
        input_values (torch.Tensor): The input data as numerical values to be encoded to population codes
        out_features (int): The number of output *per* input value
        scale (torch.Tensor): The scaling factor for the kernels. Defaults to the maximum value of the input.
                              Can also be set for each individual sample.
        kernel: A function that takes two inputs and returns a tensor. The two inputs represent the center value
                (which changes for each index in the output tensor) and the actual data value to encode respectively.z
                Defaults to gaussian radial basis kernel function.
        distance_function: A function that calculates the distance between two numbers. Defaults to euclidean.

    Returns:
        A tensor with an extra dimension of size `seq_length` containing population encoded values of the input stimulus.
        Note: An extra step is required to convert the values to spikes, see above.
    """
    # Thanks to: https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py
    size = input_values.shape + (out_features,)
    if not scale:
        scale = input_values.max()
    centres = torch.linspace(0, scale, out_features).expand(size)
    x = input_values.unsqueeze(-1).expand(size)
    distances = distance_function(x, centres) * scale
    return kernel(distances)


def poisson_encode(
    input_values: torch.Tensor,
    seq_length: int,
    f_max: float = 100,
    dt: float = 0.001,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Encodes a tensor of input values, which are assumed to be in the
    range [0,1] into a tensor of one dimension higher of binary values,
    which represent input spikes.

    See for example https://www.cns.nyu.edu/~david/handouts/poisson.pdf.

    Parameters:
        input_values (torch.Tensor): Input data tensor with values assumed to be in the interval [0,1].
        sequence_length (int): Number of time steps in the resulting spike train.
        f_max (float): Maximal frequency (in Hertz) which will be emitted.
        dt (float): Integration time step (should coincide with the integration time step used in the model)
        generator (torch.Generator): Generator for pseudorandom numbers. Usually, generator.manual_seed(seed value) is passed as the argument

    Returns:
        A tensor with an extra dimension of size `seq_length` containing spikes (1) or no spikes (0).
    """
    return (
        torch.rand(
            seq_length,
            *input_values.shape,
            device=input_values.device,
            generator=generator,
        ).float()
        < dt * f_max * input_values
    ).float()


def poisson_encode_step(
    input_values: torch.Tensor,
    f_max: float = 1000,
    dt: float = 0.001,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Encodes a tensor of input values, which are assumed to be in the
    range [0,1] into a tensor of binary values,
    which represent input spikes.

    See for example https://www.cns.nyu.edu/~david/handouts/poisson.pdf.

    Parameters:
        input_values (torch.Tensor): Input data tensor with values assumed to be in the interval [0,1].
        f_max (float): Maximal frequency (in Hertz) which will be emitted.
        dt (float): Integration time step (should coincide with the integration time step used in the model)

    Returns:
        A tensor containing binary values in .
    """
    return (
        torch.rand(
            *input_values.shape, device=input_values.device, generator=generator
        ).float()
        < dt * f_max * input_values
    ).float()


def signed_poisson_encode(
    input_values: torch.Tensor,
    seq_length: int,
    f_max: float = 100,
    dt: float = 0.001,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Encodes a tensor of input values, which are assumed to be in the
    range [-1,1] into a tensor of one dimension higher of binary values,
    which represent input spikes.

    Parameters:
        input_values (torch.Tensor): Input data tensor with values assumed to be in the interval [-1,1].
        sequence_length (int): Number of time steps in the resulting spike train.
        f_max (float): Maximal frequency (in Hertz) which will be emitted.
        dt (float): Integration time step (should coincide with the integration time step used in the model)
        generator (torch.Generator): Generator for pseudorandom numbers. Usually, generator.manual_seed(seed value) is passed as the argument

    Returns:
        A tensor with an extra dimension of size `seq_length` containing values in {-1,0,1}
    """
    return (
        torch.sign(input_values)
        * (
            torch.rand(seq_length, *input_values.shape, generator=generator).float()
            < dt * f_max * torch.abs(input_values)
        ).float()
    )


def signed_poisson_encode_step(
    input_values: torch.Tensor,
    f_max: float = 1000,
    dt: float = 0.001,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Creates a poisson distributed signed spike vector, when

    Parameters:
        input_values (torch.Tensor): Input data tensor with values assumed to be in the interval [-1,1].
        f_max (float): Maximal frequency (in Hertz) which will be emitted.
        dt (float): Integration time step (should coincide with the integration time step used in the model)
        generator (torch.Generator): Generator for pseudorandom numbers. Usually, generator.manual_seed(seed value) is passed as the argument

    Returns:
        A tensor containing values in {-1,0,1}.
    """
    return (
        torch.sign(input_values)
        * (
            torch.rand(
                *input_values.shape, device=input_values.device, generator=generator
            ).float()
            < dt * f_max * torch.abs(input_values)
        ).float()
    )


def spike_latency_lif_encode(
    input_current: torch.Tensor,
    seq_length: int,
    p: LIFParameters = LIFParameters(),
    dt=0.001,
) -> torch.Tensor:
    """Encodes an input value by the time the first spike occurs.
    Similar to the ConstantCurrentLIFEncoder, but the LIF can be
    thought to have an infinite refractory period.

    Parameters:
        input_current (torch.Tensor): Input current to encode (needs to be positive).
        sequence_length (int): Number of time steps in the resulting spike train.
        p (LIFParameters): Parameters of the LIF neuron model.
        dt (float): Integration time step (should coincide with the integration time step used in the model)
    """
    voltage = torch.zeros_like(input_current)
    z = torch.zeros_like(input_current)
    mask = torch.zeros_like(input_current)
    zeros_mask = torch.zeros_like(input_current)
    spikes = []

    for _ in range(seq_length):
        z, voltage = lif_current_encoder(
            input_current=input_current, voltage=voltage, p=p, dt=dt
        )
        spikes.append(torch.where(mask > 0, zeros_mask, z))
        mask += z

    return torch.stack(spikes)


def spike_latency_encode(input_spikes: torch.Tensor) -> torch.Tensor:
    """
    For all neurons, remove all but the first spike. This encoding basically measures the time it takes for a
    neuron to spike *first*. Assuming that the inputs are constant, this makes sense in that strong inputs spikes
    fast.

    See `R. Van Rullen & S. J. Thorpe (2001): Rate Coding Versus Temporal Order Coding: What the Retinal Ganglion Cells Tell the Visual Cortex <https://doi.org/10.1162/08997660152002852>`_.

    Spikes are identified by their unique position within each sequence.

    Example:
        >>> data = torch.as_tensor([[0, 1, 1], [1, 1, 1]])
        >>> spike_latency_encode(data)
        tensor([[0, 1, 1],
                [1, 0, 0]])

    Parameters:
        input_spikes (torch.Tensor): A tensor of input spikes, assumed to be at least 2D (sequences, ...)

    Returns:
        A tensor where the first spike (1) is retained in the sequence
    """
    mask = torch.zeros(input_spikes.size()[1:], device=input_spikes.device)
    spikes = []
    zero_spikes = torch.zeros_like(input_spikes[0])
    for index in range(input_spikes.shape[0]):
        spikes.append(torch.where(mask > 0, zero_spikes, input_spikes[index]))
        mask += spikes[-1]
    return torch.stack(spikes)
