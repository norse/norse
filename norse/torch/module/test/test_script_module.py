from typing import Tuple
import torch
from norse.torch import LIFFeedForwardState
from norse.torch.functional.lif import _lif_feed_forward_step_jit, LIFParametersJIT


class LiftJ(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super(LiftJ, self).__init__()
        self.lifted_module = module

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:

        T = x.shape[0]
        outputs = []

        for ts in range(T):
            out = self.lifted_module(x[ts])
            outputs += [out]

        return torch.stack(outputs)

class LIFJ(torch.nn.Module):
    def __init__(
        self,
        p: LIFParametersJIT = LIFParametersJIT(),
        dt: float = 0.001
    ):
        super().__init__()
        self.p = p
        self.dt = dt

    def extra_repr(self) -> str:
        return f"p={self.p}, dt={self.dt}"

    def forward(self, input_tensor: torch.Tensor):
        T = input_tensor.shape[0]
        outputs = []
        state = LIFFeedForwardState(
            v = torch.zeros(input_tensor.shape[1:], device=input_tensor.device),
            i = torch.zeros(input_tensor.shape[1:], device=input_tensor.device)
        )

        for ts in range(T):
            out, state = _lif_feed_forward_step_jit(
                input_tensor[ts],
                state,
                LIFParametersJIT(*self.p),
                self.dt,
            )
            outputs.append(out)

        return torch.stack(outputs) #, state

def test_script_module():
    model = torch.nn.Sequential(
        LiftJ(torch.nn.Conv2d(1, 20, 5, 1)),
        LIFJ(),
        LiftJ(torch.nn.Conv2d(20, 50, 5, 1)),
        LIFJ(),
        LiftJ(torch.nn.Flatten()),
        LiftJ(torch.nn.Linear(800, 10))
    )

    torch.jit.script(model)
