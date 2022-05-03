from typing import Iterator
import torch
from torch.utils.data.sampler import Sampler
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')

class ReSampler(Sampler):
    def __init__(self, dataset: torch.utils.data.Dataset, num_repeat:int, **kwargs):
        r""" A sampler for repeating sample

        dataset: `torch.utils.data.Dataset`
            dataset, must have classes_dict and collate_fn attributes
        """

        self.num_repeat = num_repeat
        self.dataset = dataset
        self.num_samples = len(dataset) * self.num_repeat

    def __iter__(self) -> Iterator[int]:
        rand_tensor = []
        for i in range(len(self.dataset)):
            for _ in range(self.num_repeat):
                rand_tensor.append(i)
        # Repeat
        yield from iter(rand_tensor)

    def __len__(self) -> int:
        return self.num_samples
