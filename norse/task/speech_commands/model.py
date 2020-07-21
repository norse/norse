import torch
from norse.torch.module.lif import LIFCell
from norse.torch.module.lsnn import LSNNCell
from norse.torch.module.leaky_integrator import LICell


class SNNModel(torch.nn.Module):
    def __init__(self, cell, n_features=128, n_input=80, n_output=10):
        super(LIFModel, self).__init__()
        self.n_features = n_features
        self.n_input = n_input
        self.n_output = n_output
        self.cell = cell(self.n_input, self.n_features)
        self.readout = LICell(self.n_features, self.n_output)

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        s = self.cell.initial_state(batch_size, x.device, x.dtype)
        so = self.readout.initial_state(batch_size, x.device, x.dtype)

        for ts in range(seq_length):
            z, s = self.cell(x[ts, :], s)
            v, so = self.readout(z, so)

        x = torch.nn.functional.log_softmax(v, dim=1)
        return x


def lsnn_model(n_features=128, n_input=80, n_output=10):
    return SNNModel(LSNNCell, n_features, n_input, n_output)


def lif_model(n_features=128, n_input=80, n_output=10):
    return SNNModel(LIFCell, n_features, n_input, n_output)


class LSTMModel(torch.nn.Module):
    def __init__(self, n_features=128, n_input=80, n_output=10):
        super(LSTMModel, self).__init__()
        self.n_features = n_features
        self.n_input = n_input
        self.n_output = n_output
        self.lstm = torch.nn.LSTM(self.n_input, self.n_features)
        self.readout = torch.nn.Linear(self.n_features, self.n_output)

    def forward(self, x):
        _, (x, _) = self.lstm(x)
        x = self.readout(x.squeeze(0))
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x
