import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class DistributedDataloader(DataLoader):
    def __init__(self, dataset, local_rank=0, **kwargs) -> None:
        self.dataset = dataset

        self.sampler = DistributedSampler(dataset, rank=local_rank, shuffle=True)

        # To re-seed the randomness everytime we start a worker
        def worker_init_fn(worker_id): 
            return np.random.seed(torch.initial_seed()%(2**31) + worker_id + local_rank*100)

        super().__init__(
          dataset=dataset, 
          sampler=self.sampler,
          worker_init_fn=worker_init_fn,
          **kwargs)