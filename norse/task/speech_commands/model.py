import torch
from norse.torch.module.lif import LIFCell
from norse.torch.module.leaky_integrator import LICell


class LIFModel(torch.nn.Module):
    def __init__(self, n_features=128, n_input=80, n_output=10):
        super(LIFModel, self).__init__()
        self.n_features = n_features
        self.n_input = n_input
        self.n_output = n_output
        self.lif = LIFCell(self.n_input, self.n_features)
        self.readout = LICell(self.n_features, self.n_output)

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        s = self.lif.initial_state(batch_size, x.device, x.dtype)
        so = self.readout.initial_state(batch_size, x.device, x.dtype)

        for ts in range(seq_length):
            z, s = self.lif(x[ts, :], s)
            v, so = self.readout(z, so)

        x = torch.nn.functional.log_softmax(v, dim=1)
        return x


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
