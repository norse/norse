"""
Threshold modules for spiking activations.
"""

import torch
import norse


class SpikeThreshold(torch.nn.Module):

    def __init__(
        self,
        threshold: torch.Tensor = torch.tensor([1.0]),
        method: str = "super",
        alpha: float = 100.0,
    ):
        """
        Wraps the threshold function in a torch.nn.Module.

        Args:
            threshold (torch.Tensor): threshold value for the spike function. Defaults to 1.0.
            method (str): method to determine the spike threshold. Defaults to SuperSpike "super".
            alpha (float): hyper parameter to use in surrogate gradient computation, defaults to 100.0.
        """
        super().__init__()
        self.alpha = torch.as_tensor(alpha)
        self.threshold = threshold
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return norse.torch.functional.threshold.threshold(
            x - self.threshold, method=self.method, alpha=self.alpha
        )
