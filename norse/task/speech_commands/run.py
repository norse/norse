import torch

# pytype: disable=import-error
import torchaudio

# pytype: enable=import-error
import argparse

from norse.dataset.speech_commands import SpeechCommandsDataset, prepare_dataset

# pytype: disable=import-error
from norse.task.speech_commands.model import LSTMModel, lsnn_model, lif_model

# pytype: enable=import-error

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--batch_size", default=16)
parser.add_argument("--device", default="cuda")
parser.add_argument("--model", default="lif")

args = parser.parse_args()

BATCH_SIZE = args.batch_size  # 16
LR = args.learning_rate  # 0.0001
DEVICE = args.device  # "cuda"
MODEL = args.model  # "lif"

speech_commands = torchaudio.datasets.SPEECHCOMMANDS(root=".", download=True)
train_sc, valid_sc, test_sc = prepare_dataset(speech_commands)


train_transform = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35),
)

valid_transform = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),
)

train_speech_commands = SpeechCommandsDataset(
    dataset=train_sc, transform=train_transform
)
valid_speech_commands = SpeechCommandsDataset(
    dataset=valid_sc, transform=valid_transform
)

# pytype: disable=module-attr
train_loader = torch.utils.data.DataLoader(
    train_speech_commands, batch_size=BATCH_SIZE, shuffle=True
)

valid_loader = torch.utils.data.DataLoader(
    valid_speech_commands, batch_size=BATCH_SIZE, shuffle=True
)
# pytype: enable=module-attr

if MODEL == "lif":
    model = lif_model(n_output=13).to(DEVICE)
elif MODEL == "lsnn":
    model = lsnn_model(n_output=13).to(DEVICE)
else:
    model = LSTMModel(n_output=13).to(DEVICE)

loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def test(model, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            data = data.squeeze(1)
            data = data.permute(2, 0, 1)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)"
    )

    return test_loss, accuracy


for epoch in range(10):
    model.train()
    print(f"=========== {epoch} ===========")
    for idx, (data, target) in enumerate(train_loader):
        data = data.squeeze(1)
        data = data.permute(2, 0, 1)
        model.zero_grad()
        out = model(data.to(DEVICE))
        loss = loss_function(out, target.to(DEVICE))
        loss.backward()
        optimizer.step()
        if idx % 1000 == 0:
            print(f"{idx} {loss.data}")

    test(model, valid_loader, epoch)
