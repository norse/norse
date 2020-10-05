import torch
from norse.torch.module.lif import LIFCell

def test_training_lif():
    data = torch.ones(10, 32) # 10 timesteps, 32 neurons
    # model = torch.nn.Sequential(

    # )