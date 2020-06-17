import torch


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
