import torch
import numpy as np

from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif_correlation import LIFCorrelation
from norse.torch.functional.correlation_sensor import correlation_based_update


def test_lif_correlation_training():
    def generate_random_data(
        seq_length,
        batch_size,
        input_features,
        device="cpu",
        dtype=torch.float,
        dt=0.001,
    ):
        freq = 5
        prob = freq * dt
        mask = torch.rand(
            (seq_length, batch_size, input_features), device=device, dtype=dtype
        )
        x_data = torch.zeros(
            (seq_length, batch_size, input_features),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        x_data[mask < prob] = 1.0
        y_data = torch.tensor(1 * (np.random.rand(batch_size) < 0.5), device=device)
        return x_data, y_data

    seq_length = 50
    batch_size = 1
    input_features = 10
    hidden_features = 8
    output_features = 2

    device = "cpu"

    x, y_data = generate_random_data(
        seq_length=seq_length,
        batch_size=batch_size,
        input_features=input_features,
        device=device,
    )

    input_weights = (
        torch.randn((input_features, hidden_features), device=device).float().t()
    )

    recurrent_weights = torch.randn(
        (hidden_features, hidden_features), device=device
    ).float()

    lif_correlation = LIFCorrelation(input_features, hidden_features)
    out = LILinearCell(hidden_features, output_features).to(device)
    log_softmax_fn = torch.nn.LogSoftmax(dim=1)
    loss_fn = torch.nn.NLLLoss()

    linear_update = torch.nn.Linear(2 * 10 * 8, 10 * 8)
    rec_linear_update = torch.nn.Linear(2 * 8 * 8, 8 * 8)

    optimizer = torch.optim.Adam(
        list(linear_update.parameters())
        + [input_weights, recurrent_weights]
        + list(out.parameters()),
        lr=1e-1,
    )

    loss_hist = []
    num_episodes = 3

    for e in range(num_episodes):
        s1 = None
        so = None

        voltages = torch.zeros(seq_length, batch_size, output_features, device=device)
        hidden_voltages = torch.zeros(
            seq_length, batch_size, hidden_features, device=device
        )
        hidden_currents = torch.zeros(
            seq_length, batch_size, hidden_features, device=device
        )

        optimizer.zero_grad()

        for ts in range(seq_length):
            z1, s1 = lif_correlation(
                x[ts, :, :],
                input_weights=input_weights,
                recurrent_weights=recurrent_weights,
                state=s1,
            )

            input_weights = correlation_based_update(
                ts,
                linear_update,
                input_weights.detach(),
                s1.input_correlation_state,
                0.01,
                10,
            )
            recurrent_weights = correlation_based_update(
                ts,
                rec_linear_update,
                recurrent_weights.detach(),
                s1.recurrent_correlation_state,
                0.01,
                10,
            )
            vo, so = out(z1, so)
            hidden_voltages[ts, :, :] = s1.lif_state.v.detach()
            hidden_currents[ts, :, :] = s1.lif_state.i.detach()
            voltages[ts, :, :] = vo

        m, _ = torch.max(voltages, dim=0)

        log_p_y = log_softmax_fn(m)
        loss_val = loss_fn(log_p_y, y_data.long())

        loss_val.backward()
        optimizer.step()
        loss_hist.append(loss_val.item())
        print(f"{e}/{num_episodes}: {loss_val.item()}")

    # assert loss_hist[0] > loss_hist[1]
    # assert loss_hist[1] > loss_hist[2]
