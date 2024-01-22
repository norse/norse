"""This module provides a wrapper around the torchaudio.SPEECHCOMMANDS dataset.
"""

import torch

# pytype: disable=import-error
import torchaudio

# pytype: enable=import-error
import os
import glob


def generate_background_noise(speech_commands):
    """Split the background noise provided by the dataset in 1 second chunks.

    Parameters:
        speech_commands (torch.utils.data.Dataset): Speech Command dataset as defined by torchaudio.
    """
    background_noise = glob.glob(
        os.path.join(speech_commands._path, "_background_noise_", "*.wav")
    )
    os.makedirs(os.path.join(speech_commands._path, "background"), exist_ok=True)

    for file in background_noise:
        waveform, sample_rate = torchaudio.load(file)
        background_waveforms = torch.split(waveform, sample_rate, dim=1)[:-1]

        for idx, background_waveform in enumerate(background_waveforms):
            torchaudio.save(
                os.path.join(
                    speech_commands._path,
                    "background",
                    f"{hash(waveform)}_nohash_{idx}.wav",
                ),
                background_waveform,
                sample_rate=sample_rate,
            )


def prepare_dataset(speech_commands):
    """Prepare the speech command dataset by splitting it into training, validation
    and testing according to the method described in "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition"
    (https://arxiv.org/abs/1804.03209). It is meant to be used with the SPEECHCOMMANDS
    dataset provided by torchaudio.

    .. code-block:: python
        speech_commands = torchaudio.datasets.SPEECHCOMMANDS(root='.', download=True)
        train_sc, valid_sc, test_sc = prepare_dataset(speech_commands)

    Parameters:
        speech_commands (torch.utils.data.Dataset): Speech Commands dataset as defined by torchaudio.
    """
    with open(os.path.join(speech_commands._path, "validation_list.txt")) as f:
        text = f.read()
        valid_filelist = text.split("\n")
        valid_filelist = [
            os.path.join(speech_commands._path, p) for p in valid_filelist
        ]

    with open(os.path.join(speech_commands._path, "testing_list.txt")) as f:
        text = f.read()
        test_filelist = text.split("\n")
        test_filelist = [os.path.join(speech_commands._path, p) for p in test_filelist]

    train_sc = torchaudio.datasets.SPEECHCOMMANDS(root=".")
    train_sc._walker = list(
        filter(
            lambda w: w not in valid_filelist and w not in test_filelist,
            train_sc._walker,
        )
    )

    valid_sc = torchaudio.datasets.SPEECHCOMMANDS(root=".")
    valid_sc._walker = list(filter(lambda w: w in valid_filelist, valid_sc._walker))

    test_sc = torchaudio.datasets.SPEECHCOMMANDS(root=".")
    test_sc._walker = list(filter(lambda w: w in valid_filelist, valid_sc._walker))

    # no generate the background noise category
    generate_background_noise(speech_commands)

    ratios = [
        len(train_sc._walker) / len(speech_commands._walker),
        len(valid_sc._walker) / len(speech_commands._walker),
        len(test_filelist) / len(speech_commands._walker),
    ]

    files = glob.glob(os.path.join(speech_commands._path, "background/*.wav"))
    train_idx = int(ratios[0] * len(files))
    valid_idx = train_idx + int(ratios[1] * len(files))
    test_idx = valid_idx + int(ratios[2] * len(files))
    training = files[:train_idx]
    valid = files[train_idx:valid_idx]
    test = files[valid_idx : (test_idx + 1)]

    train_sc._walker += training
    test_sc._walker += test
    valid_sc._walker += valid

    return train_sc, valid_sc, test_sc


def label_to_index(dataset):
    """Generate integer labels for each of the classes in the speech commands dataset.

    Parameters:
        speech_commands (torch.utils.data.Dataset): Speech Commands dataset as defined by torchaudio.
    """
    known = {
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
        "background",
    }
    all_labels = set()

    for _, _, label, _, _ in dataset:
        all_labels.add(label)

    unknown = all_labels - known
    label_to_index = {
        **{key: idx for idx, key in enumerate(known)},
        **{key: 12 for key in unknown},
    }
    return label_to_index


class SpeechCommandsDataset(torch.utils.data.Dataset):
    """Speech Commands dataset as described in "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition"
    (https://arxiv.org/abs/1804.03209). This is meant as a wrapper around the corresponding
    SPEECHCOMMANDS dataset defined in torchaudio.

    .. code-block:: python
        speech_commands = torchaudio.datasets.SPEECHCOMMANDS(root='.', download=True)
        train_sc, valid_sc, test_sc = prepare_dataset(speech_commands)

        train_transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=81),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )

        valid_transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=81),
        )

        train_speech_commands = SpeechCommands(dataset=train_sc, transform=train_transform)
        valid_speech_commands = SpeechCommands(dataset=valid_sc, transform=valid_transform)

    Parameters:
        speech_commands (torch.utils.data.Dataset): Speech Commands dataset as defined by torchaudio.
        transform (torch.nn.Module): Sequence of transformations to apply to waveform data.
    """

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        self.label_to_index = label_to_index(dataset)

    def __getitem__(self, index):
        (waveform, sample_rate, label, _, _) = self.dataset[index]

        padding = int((sample_rate - waveform.shape[1]))
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        spec = self.transform(waveform)
        idx = self.label_to_index[label]

        return spec, idx

    def __len__(self):
        return len(self.dataset)
