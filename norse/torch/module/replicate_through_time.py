import torch

class ReplicateThroughTime(torch.nn.Module):
    def __init__(self, module, T):
        self.module_list = torch.nn.ModuleList([module for _ in range(T)])
        self.T = T

    def forward(self, x):
        outputs = []
        for ts in range(self.T):
            y = self.module_list[ts](x[ts])
            outputs.append(y)

        return torch.stack(outputs)