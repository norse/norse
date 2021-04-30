"""
The Spiking Heidelberg Digits (SHD) audio dataset
https://compneuro.net/posts/2019-spiking-heidelberg-digits/
Licensed under CC A 4.0

Cramer, B., Stradmann, Y., Schemmel, J., and Zenke, F. (2019).
The Heidelberg spiking datasets for the systematic evaluation of spiking neural networks.
ArXiv:1910.07407 [Cs, q-Bio]. https://arxiv.org/abs/1910.07407
"""

from itertools import chain
import os

# pytype: disable=import-error
import h5py
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

# pytype: enable=import-error
import torch


class SpikingHeidelbergDigitsDataset(torch.utils.data.IterableDataset):
    """
    Initialises, but does not download by default, the
    `Spiking Heidelberg audio dataset <https://compneuro.net/posts/2019-spiking-heidelberg-digits/>`_.

    Parameters:
        root (str): The root of the dataset directory
        sparse (bool): Whether or not to load the data as sparse tensors. True by default
        train (bool, optional): If True, creates dataset from training set, otherwise
            we only use te test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    test_file, test_file_md5 = "shd_test.h5", "207ad1295a0ab611e22d1734989c254e"
    test_url = "https://compneuro.net/datasets/shd_test.h5.gz"
    test_checksum = "3062a80ec0c5719404d5b02e166543b1"
    train_file, train_file_md5 = "shd_train.h5", "8e9877c85f29a28dd353ea6a54402667"
    train_url = "https://compneuro.net/datasets/shd_train.h5.gz"
    train_checksum = "d47c9825dee33347913e8ce0f2be08b0"
    max_length = 1.4  # Max dataset length in seconds
    n_units = 700  # 700 channels

    def __init__(self, root, dt=0.001, sparse=True, train=True, download=False):
        super().__init__()
        self.root = root
        self.train = train
        self.dt = dt

        if download:
            self.download()

        self.train_fp = (
            None
            if not self.train
            else h5py.File(os.path.join(self.root, self.train_file), "r")
        )
        self.test_fp = h5py.File(os.path.join(self.root, self.test_file), "r")

    def _check_integrity(self):
        return check_integrity(
            os.path.join(self.root, self.test_file), self.test_file_md5
        ) and (
            not self.train
            or check_integrity(
                os.path.join(self.root, self.train_file), self.train_file_md5
            )
        )

    def _bin_spikes(self, tup):
        times, units, label = tup
        assert len(times) == len(units), "Spikes and units must have same length"
        length = int(self.max_length // self.dt + 1)
        data = torch.zeros((length, self.n_units), dtype=torch.uint8)
        for index, time_value in enumerate(times):
            time_index = int(time_value // self.dt)
            unit_index = units[index]
            data[time_index][unit_index] = 1
        return data.to_sparse(), torch.as_tensor(int(label), dtype=torch.uint8)

    def download(self):
        if not self._check_integrity():
            if self.train:
                download_and_extract_archive(
                    self.train_url, self.root, md5=self.train_checksum
                )
            download_and_extract_archive(
                self.test_url, self.root, md5=self.test_checksum
            )

    def __iter__(self):
        def _data_iterator(self, fp):
            return zip(fp["spikes"]["times"], fp["spikes"]["units"], fp["labels"])

        iterator = map(self._bin_spikes, self._data_iterator(self.test_fp))
        if self.train:
            iterator = chain(
                map(self._bin_spikes, self._data_iterator(self.test_fp)), iterator
            )
        return iterator
