import math
import os
import struct
import wave

import pytest
import torch

torchaudio = pytest.importorskip("torchaudio")

from norse.dataset import speech_commands  # noqa: E402

WORDS = ("yes", "no", "up")


def _write_wav(path, seconds, sample_rate=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = int(sample_rate * seconds)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(1000 * math.sin(i / 7))) for i in range(frames)
            )
        )


def _load(path):
    with wave.open(path) as w:
        n, sample_rate = w.getnframes(), w.getframerate()
        samples = struct.unpack(f"<{n}h", w.readframes(n))
    return torch.tensor(samples, dtype=torch.float32).unsqueeze(0) / 32768, sample_rate


def _save(path, tensor, sample_rate):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(max(-1.0, min(1.0, v)) * 32767))
                for v in tensor[0].tolist()
            )
        )


@pytest.fixture
def fake_speech_commands(tmp_path, monkeypatch):
    """A tiny Speech Commands tree in the layout torchaudio expects (no download)."""
    # torchaudio >= 2.9 loads audio through torchcodec; plain WAV I/O keeps the test self-contained.
    monkeypatch.setattr(torchaudio, "load", _load)
    monkeypatch.setattr(torchaudio, "save", _save)
    base = tmp_path / "SpeechCommands" / "speech_commands_v0.02"
    files = [f"{word}/{i:08x}_nohash_0.wav" for word in WORDS for i in range(10)]
    for f in files:
        _write_wav(str(base / f), 0.1)
    _write_wav(str(base / "_background_noise_" / "white_noise.wav"), 3.5)
    valid, test = files[0::5], files[1::5]
    (base / "validation_list.txt").write_text("\n".join(valid) + "\n")
    (base / "testing_list.txt").write_text("\n".join(test) + "\n")
    dataset = torchaudio.datasets.SPEECHCOMMANDS(root=str(tmp_path), download=False)
    return dataset, base, set(files) - set(valid) - set(test), set(valid), set(test)


def _relative(dataset, base):
    return {
        os.path.relpath(os.path.normpath(w), base).replace(os.sep, "/")
        for w in dataset._walker
        if "background" not in w
    }


def test_prepare_dataset_splits_match_the_official_lists(
    fake_speech_commands, monkeypatch, tmp_path
):
    dataset, base, train, valid, test = fake_speech_commands
    monkeypatch.chdir(tmp_path)
    train_sc, valid_sc, test_sc = speech_commands.prepare_dataset(dataset)
    assert _relative(train_sc, base) == train
    assert _relative(valid_sc, base) == valid
    assert _relative(test_sc, base) == test


def test_prepare_dataset_uses_the_given_dataset_root(
    fake_speech_commands, monkeypatch, tmp_path
):
    dataset, base, train, *_ = fake_speech_commands
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # the working directory is not the dataset root
    train_sc, _, _ = speech_commands.prepare_dataset(dataset)
    assert _relative(train_sc, base) == train


def test_prepare_dataset_does_not_accumulate_background_files(
    fake_speech_commands, monkeypatch, tmp_path
):
    dataset, base, *_ = fake_speech_commands
    monkeypatch.chdir(tmp_path)
    speech_commands.prepare_dataset(dataset)
    first = sorted(os.listdir(base / "background"))
    speech_commands.prepare_dataset(dataset)
    assert sorted(os.listdir(base / "background")) == first
    assert len(first) == 3  # 3.5 s of noise -> three whole 1 s chunks
