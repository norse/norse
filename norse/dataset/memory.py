<<<<<<< HEAD
<<<<<<< HEAD
from typing import Optional

import torch
import torch.utils.data
=======
=======
from typing import Optional

>>>>>>> 3a3ea38... Added seed parameter for the memory dataset
import torch
<<<<<<< HEAD
>>>>>>> 267ca63... Added memory dataset and reworked memory task
=======
import torch.utils.data
>>>>>>> 4f2e8da... Fixed pytype errors

from norse.torch.functional.encode import poisson_encode


class MemoryStoreRecallDataset(torch.utils.data.Dataset):
    """
    A memory dataset that generates random patterns of 4-bit data, and
    a 2-bit command pattern (store and recall).

    Note that you can control the randomness by setting `a manual seed in
    PyTorch <https://pytorch.org/docs/stable/generated/torch.manual_seed.html>`_.

    Inspired by Bellec et al.: `Biologically inspired alternatives to backpropagation through time for learning in recurrent neural nets <https://arxiv.org/abs/1901.09049>`_.

    Arguments:
        samples (int): Number of samples in the dataset.
        seq_length (int): Number of timesteps to simulate per command. Defaults to 100.
        seq_periods (int): Number of commands in one sample. Defaults to 12.
        seq_repetitions (int): Number of times one store/recall pair occurs in a single sample. Defaults to 4.
        population_size (int): Number of neurons encoding each command. Defaults to 5.
        poisson_rate (int): Poisson rate for each command in Hz. Defaults to 250.
        dt (float): Timestep for the dataset. Defaults to 0.001 (1000Hz).
<<<<<<< HEAD
<<<<<<< HEAD
        seed (Optional[int]): Optional seed for the random generator
=======
>>>>>>> 267ca63... Added memory dataset and reworked memory task
=======
        seed (Optional[int]): Optional seed for the random generator
>>>>>>> 3a3ea38... Added seed parameter for the memory dataset
    """

    def __init__(
        self,
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 3a3ea38... Added seed parameter for the memory dataset
        samples: int,
        seq_length: int = 100,
        seq_periods: int = 12,
        seq_repetitions: int = 4,
        population_size: int = 5,
        poisson_rate: int = 100,
        dt: float = 0.001,
        seed: Optional[int] = None,
<<<<<<< HEAD
=======
        samples,
        seq_length=100,
        seq_periods=12,
        seq_repetitions=4,
        population_size=5,
        poisson_rate=100,
        dt=0.001,
>>>>>>> 267ca63... Added memory dataset and reworked memory task
=======
>>>>>>> 3a3ea38... Added seed parameter for the memory dataset
    ):
        self.samples = samples
        self.seq_length = seq_length
        self.seq_periods = seq_periods
        self.seq_repetitions = seq_repetitions
        self.population_size = population_size
        self.poisson_rate = poisson_rate
        self.dt = dt

        self.store_indices = torch.randint(
            low=0,
            high=seq_periods // 2,
            size=(samples, seq_repetitions),
        )
        self.recall_indices = torch.randint(
            low=seq_periods // 2,
            high=seq_periods,
            size=(samples, seq_repetitions),
        )
<<<<<<< HEAD
<<<<<<< HEAD
        self.generator = None if seed is None else torch.manual_seed(seed)
=======
>>>>>>> 267ca63... Added memory dataset and reworked memory task
=======
        self.generator = None if seed is None else torch.manual_seed(seed)
>>>>>>> 3a3ea38... Added seed parameter for the memory dataset

    def __len__(self):
        return self.samples

    def _generate_sequence(self, idx, rep_idx):
        data_pattern = torch.stack(
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 3a3ea38... Added seed parameter for the memory dataset
            [
                torch.randperm(2, generator=self.generator)
                for _ in range(self.seq_periods)
            ]
<<<<<<< HEAD
=======
            [torch.randperm(2) for _ in range(self.seq_periods)]
>>>>>>> 267ca63... Added memory dataset and reworked memory task
=======
>>>>>>> 3a3ea38... Added seed parameter for the memory dataset
        ).byte()
        store_index = self.store_indices[idx][rep_idx]
        recall_index = self.recall_indices[idx][rep_idx]
        store_pattern = torch.zeros((self.seq_periods, 1)).byte()
        recall_pattern = store_pattern.clone()
        label_pattern = torch.zeros((self.seq_periods, 2)).byte()

        store_pattern[store_index] = 1
        recall_pattern[recall_index] = 1
        label_class = data_pattern[store_index].byte()
<<<<<<< HEAD
<<<<<<< HEAD
        label_pattern[recall_index] = label_class
        data_pattern[recall_index] = torch.zeros(2)

        def encode_pattern(pattern, hz):
            return poisson_encode(
                pattern.repeat_interleave(self.population_size, dim=1),
                seq_length=self.seq_length,
                f_max=hz,
                dt=self.dt,
            )

        encoded_data_pattern = encode_pattern(data_pattern, self.poisson_rate)
        encoded_command_pattern = encode_pattern(
            torch.cat((store_pattern, recall_pattern), dim=1), self.poisson_rate // 2
        )
        encoded_pattern = torch.cat(
            (encoded_data_pattern, encoded_command_pattern), dim=2
        )
        encoded = torch.cat(encoded_pattern.chunk(self.seq_periods, dim=1)).squeeze()
=======
        label_pattern[store_index] = label_class
        label_pattern[recall_index] = label_class
        data_pattern[recall_index] = torch.zeros(2)

        input_pattern = torch.cat((data_pattern, store_pattern, recall_pattern), dim=1)
        input_pattern = input_pattern.repeat_interleave(self.population_size, dim=1)
        encoded = poisson_encode(
            input_pattern,
            seq_length=self.seq_length,
            f_max=self.poisson_rate,
            dt=self.dt,
        )
        encoded = torch.cat(encoded.chunk(self.seq_periods, dim=1)).squeeze()
>>>>>>> 267ca63... Added memory dataset and reworked memory task
=======
        label_pattern[recall_index] = label_class
        data_pattern[recall_index] = torch.zeros(2)

        def encode_pattern(pattern, hz):
            return poisson_encode(
                pattern.repeat_interleave(self.population_size, dim=1),
                seq_length=self.seq_length,
                f_max=hz,
                dt=self.dt,
            )

        encoded_data_pattern = encode_pattern(data_pattern, self.poisson_rate)
        encoded_command_pattern = encode_pattern(
            torch.cat((store_pattern, recall_pattern), dim=1), self.poisson_rate // 2
        )
        encoded_pattern = torch.cat(
            (encoded_data_pattern, encoded_command_pattern), dim=2
        )
        encoded = torch.cat(encoded_pattern.chunk(self.seq_periods, dim=1)).squeeze()
>>>>>>> 4697ac2... Corrected memory task
        return encoded, label_pattern

    def __getitem__(self, idx):
        repetitions = [
            self._generate_sequence(idx, i) for i in range(self.seq_repetitions)
        ]
        return (
            torch.cat([rep[0] for rep in repetitions]),
            torch.cat([rep[1] for rep in repetitions]),
        )
