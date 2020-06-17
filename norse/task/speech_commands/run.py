import torch
import torchaudio
from norse.dataset.speech_commands import SpeechCommands, prepare_dataset
from norse.task.speech_commands.model import LSTMModel


BATCH_SIZE = 16
LR = 0.0001


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

train_speech_commands = SpeechCommands(dataset=train_sc, transform=train_transform)
valid_speech_commands = SpeechCommands(dataset=valid_sc, transform=valid_transform)

train_loader = torch.utils.data.DataLoader(
    train_speech_commands, batch_size=BATCH_SIZE, shuffle=True
)


valid_loader = torch.utils.data.DataLoader(
    valid_speech_commands, batch_size=BATCH_SIZE, shuffle=True
)

model = LSTMModel(n_output=13)
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def test(model, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
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
    print(f"=========== {epoch} ===========")
    for idx, (data, target) in enumerate(train_loader):
        data = data.squeeze(1)
        data = data.permute(2, 0, 1)
        model.zero_grad()
        out = model(data)
        loss = loss_function(out, target)
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print(f"{idx} {loss.data}")

    test(model, valid_loader, epoch)
