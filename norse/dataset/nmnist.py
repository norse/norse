import logging
import os
import pathlib

import torch


from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
)


class NMNIST(torch.utils.data.Dataset):
    """
    The IBM gesture dataset containing 11 hand gestures from 29 subjects under
    3 illumination conditions recorded from a DVS128 camera

    Unless otherwise specified, the dataset will be downloaded to the default
    (or given) path

    **Depends** on the `AEDAT <https://github.com/norse/aedat>`_ library

    Source: http://www.research.ibm.com/dvsgesture/

    Parameters:
        root (str): The root of the dataset directory
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    tensor_size = (10000000, 128, 128)
    zip_md5 = "c5b12b1213584bd3fe976b55fe43c835"
    filename = "nmnist.zip"
    filename_train = "Train.zip"
    filename_test = "Test.zip"
    data_dir = "nmnist"
    url = "https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABrCc6FewNZSNsoObWJqY74a"

    def __init__(self, root, train=True, download=False):
        super().__init__()

        self.root = root
        self.train = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        logging.info("NMNIST dataset downloaded and verified")

        self.current_list = self.train_list if self.train else self.test_list
        self.current_files = [
            (
                os.path.join(self.root, self.data_dir, x[1]),
                (os.path.join(self.root, self.data_dir, f"{x[1][:-6]}_labels.csv")),
            )
            for x in self.current_list
        ]

        for event_file, label_file in self.current_files:
            self.data.load(event_file, label_file)

    def _check_integrity(self):
        for fentry in self.train_list + self.test_list:
            md5, filename = fentry[0], fentry[1]
            fpath = os.path.join(self.root, self.data_dir, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.zip_md5
        )
        test_path = pathlib.Path(self.root) / self.data_dir / self.filename_test
        extract_archive(test_path, self.root)
        if self.train:
            train_path = pathlib.Path(self.root) / self.data_dir / self.filename_train
            extract_archive(train_path, self.root)

    def __getitem__(self, index):
        return (
            aedat.convert_polarity_events(
                self.data.datapoints[index].events,
                self.tensor_size,  # Force to specific width for torch.stack
            ),
            self.data.datapoints[index].label,
        )

    def __len__(self):
        return len(self.data.datapoints)
