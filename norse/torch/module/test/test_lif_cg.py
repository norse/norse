import torch

from norse.torch.module import encode
from norse.torch.module.lif import LIFCell
from norse.torch.module.lif_cg import LIFCellCG


class SNCGNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, T=10):
        super(SNCGNetwork, self).__init__()
        self.lif1 = LIFCellCG()
        self.lif2 = LIFCellCG()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=False, device="cuda")
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim, bias=False, device="cuda")

        self.encoder = encode.ConstantCurrentLIFEncoder(T)

    def forward(self, input):
        # encode input
        encoded_input = self.encoder(input)

        seq_len, batch_size, _ = encoded_input.shape

        s1 = None
        so = None

        spike_count = 0

        for t_step in range(seq_len):
            input_spikes = encoded_input[t_step, :, :]
            z, s1 = self.lif1(self.fc1(input_spikes), s1)
            zo, self.so = self.lif2(self.fc2(z), so)

            spike_count += zo

        return spike_count / seq_len


class SNNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, T=10):
        super(SNNetwork, self).__init__()
        self.lif1 = LIFCell()
        self.lif2 = LIFCell()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=False, device="cuda")
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim, bias=False, device="cuda")

        self.encoder = encode.ConstantCurrentLIFEncoder(T)

    def forward(self, input):
        # encode input
        encoded_input = self.encoder(input)

        seq_len, batch_size, _ = encoded_input.shape

        s1 = None
        so = None

        spike_count = 0

        for t_step in range(seq_len):
            input_spikes = encoded_input[t_step, :, :]
            z, s1 = self.lif1(self.fc1(input_spikes), s1)
            zo, self.so = self.lif2(self.fc2(z), so)

            spike_count += zo

        return spike_count / seq_len


def test_lif_cg_cell_feedforward():
    cell = LIFCellCG()
    data = torch.randn(5, 2, device="cuda")
    out, s = cell(data)

    for x in s:
        assert x.shape == (5, 2)
    assert out.shape == (5, 2)


def test_lif_cg_cell_match_feedforward():
    torch.seed()
    cg_cell = LIFCellCG()
    cell = LIFCell()
    data = torch.randn(5, 2, device="cuda")
    out_cg_cell, s_cg_cell = cg_cell(data)
    out_cell, s_cell = cell(data)

    for i, x in enumerate(s_cg_cell):
        assert s_cg_cell[i].shape == s_cell[i].shape
        assert (
            s_cg_cell[i] == s_cell[i]
        ).all(), f"cg state: {s_cg_cell[i]}\nLIFCell state: {s_cell[i]}"
    assert out_cg_cell.shape == out_cell.shape
    assert (
        out_cg_cell == out_cell
    ).all(), f"cg output: {out_cg_cell}\nLIFCell output: {out_cell}"

    out_cg_cell, s_cg_cell = cg_cell(data, s_cg_cell)
    out_cell, s_cell = cell(data, s_cell)

    for i, x in enumerate(s_cg_cell):
        assert s_cg_cell[i].shape == s_cell[i].shape
        assert (
            s_cg_cell[i] == s_cell[i]
        ).all(), f"cg state: {s_cg_cell[i]}\nLIFCell state: {s_cell[i]}"
    assert out_cg_cell.shape == out_cell.shape
    assert (
        out_cg_cell == out_cell
    ).all(), f"cg output: {out_cg_cell}\nLIFCell output: {out_cell}"


def test_lif_cg_feedforward_cell_backward():
    # Tests that gradient variables can be used in subsequent applications
    gradient_flow = False
    cell = LIFCellCG()
    data = torch.randn(5, 4, device="cuda")
    out, s = cell(data)
    out, _ = cell(out, s)
    loss = out.sum()
    loss.backward()
    gradient_flow = True
    assert gradient_flow


def test_lif_cg_match_feedforward_cell_backward():
    # Tests that gradient variables match behavior of normal LIFCell
    gradient_flow = False
    cg_cell = LIFCellCG()
    cell = LIFCell()
    data = torch.ones(5, 4, device="cuda", requires_grad=True)
    cg_data = torch.ones(5, 4, device="cuda", requires_grad=True)

    cg_out, cg_s = cg_cell(cg_data)
    cg_out, cg_ = cg_cell(cg_out, cg_s)
    cg_loss = cg_out.sum()

    cg_loss.backward()
    cg_data_grad = cg_data.grad

    out, s = cell(data)
    out, _ = cell(out, s)
    loss = out.sum()

    loss.backward()
    data_grad = data.grad

    assert loss == cg_loss
    assert (data_grad == cg_data_grad).all()

    gradient_flow = True
    assert gradient_flow


def test_backward_model():
    # data
    in_features = 10
    hidden_features = 20
    out_features = 1
    batch_size = 1
    data = 100 * torch.ones(batch_size, in_features, device="cuda")

    # model
    model = SNCGNetwork(in_features, hidden_features, out_features)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer.zero_grad()

    # forward
    spike_count = model(data)

    # backward
    loss = spike_count.sum()
    loss.backward()

    optimizer.step()

    # check grad_fn
    assert loss.grad_fn is not None


def test_forward_model_cg_match():
    # data
    in_features = 10
    hidden_features = 20
    out_features = 1
    batch_size = 1
    data = 100 * torch.ones(batch_size, in_features, device="cuda")

    # model
    cg_model = SNCGNetwork(in_features, hidden_features, out_features)
    model = SNNetwork(in_features, hidden_features, out_features)

    # forward
    cg_spike_count = cg_model(data)
    spike_count = model(data)

    assert cg_spike_count == spike_count
