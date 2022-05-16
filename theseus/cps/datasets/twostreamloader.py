import random
from typing import Dict, List, Tuple, Iterable
import torch
import numpy as np

from .twosampler import TwoStreamBatchSampler
from theseus.base.datasets.dataset import ConcatDataset

class TwoStreamDataLoader(torch.utils.data.DataLoader):
    """
    Two streams for dataset. Separate batch with additional `split point` 
    """
    def __init__(
        self, 
        dataset_l: torch.utils.data.Dataset, 
        dataset_u: torch.utils.data.Dataset, 
        batch_sizes: List[int],
        **kwargs) -> None:
        self.dataset_l = dataset_l
        self.dataset_u = dataset_u

        cat_dataset = ConcatDataset([dataset_l, dataset_u])
        total_length = len(dataset_l) + len(dataset_u)
        labeled_idxs = list(range(0, len(dataset_l)))
        unlabeled_idxs = list(range(len(dataset_l), total_length))
        self.batch_sizes = batch_sizes

        self.num_classes = dataset_l.num_classes
        self._encode_masks = dataset_l._encode_masks

        sampler = TwoStreamBatchSampler(
            primary_indices=labeled_idxs,
            secondary_indices=unlabeled_idxs,
            primary_batch_size=batch_sizes[0],
            secondary_batch_size=batch_sizes[1]
        )

        super().__init__(
            dataset=cat_dataset, 
            collate_fn=self.mutual_collate_fn, 
            batch_sampler=sampler,
            **kwargs
        )

    def mutual_collate_fn(self, batch):
        """
        Mutual collate
        """
        imgs = torch.cat([i['input'] for i in batch], dim=0)
        masks = torch.cat([i['target'] for i in batch if 'target' in i], dim=0)
        img_names = [i['img_name'] for i in batch]
        ori_sizes = [i['ori_size'] for i in batch]

        
        masks = self._encode_masks(masks)
        return {
            'inputs': imgs,
            'targets': masks,
            'img_names': img_names,
            'ori_sizes': ori_sizes,
            'split_pos': masks.shape[0]
        }
        
