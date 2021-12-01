r"""Exponential smoothing is rule of thumb techique for smoothing 
        time series data using the exponential window function.

          
    """

import torch
import math
from ..functional.filter import exp_filter_step

class ExpFilter(torch.nn.Module):
    r""" 
    A exponential smoothing layer. This layer is useful 
    for smoothing out the spike activity of neurons and getting an 
    output from the network.
    
    .. math::
    s(t) = x(y) + (1 - \alpha) * s(t - \Delta t),
    where smoothing factor
    .. math::
        \alpha = 1 - e^{-\Delta t * \tau_{filter_inv}}.
        
    After application, the layer returns smoothed data. 
    """
    
    def __init__(
                self, 
                input_size:int,
                output_size:int, 
                tau_filter_inv:float = 1e3,
                dt: float = 1e-3,
                input_weights: Optional[torch.Tensor] = None, 
                bias: bool = True
                ) -> None:
        super(ExpFilter, self).__init__()
        self.input_size = torch.as_tensor(input_size)
        self.output_size = torch.as_tensor(output_size)
        self.parameter = 1 - math.exp(-dt * tau_filter_inv)
        
        if input_weights is not None:
            self.input_weights = input_weights
        else: 
            k = torch.sqrt(1.0 / self.input_size)
            self.input_weights = -k + 2 * k * torch.rand(output_size, input_size) # from - sqrt(k) to sqrt(k) (like Linear layer)
        self.linear = torch.nn.Linear(input_size, output_size, bias=bias)
        with torch.no_grad():
            self.linear.weight.copy_(self.input_weights)
            
    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, output_size={self.output_size}, parameter={self.parameter}"
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = self.linear(input_tensor)
        T = input_tensor.shape[0]
        outputs = []
        outputs.append(input_tensor[0])
        
        for ts in range(T - 1):
            out = exp_filter_step(outputs[-1], input_tensor[ts + 1], self.parameter)
            outputs.append(out)
        return torch.stack(outputs)
