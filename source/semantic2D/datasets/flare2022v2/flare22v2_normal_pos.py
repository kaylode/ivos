from typing import Union, Optional, List, Dict
import os
import os.path as osp
import torch
import numpy as np
import pandas as pd
from source.semantic2D.datasets.flare2022v2.base import (
    FLARE22V2BaseCSVDataset,
)
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")

class FLARE22V2LabelledCSVPosDataset(FLARE22V2BaseCSVDataset):
    """
    Load in csv
    """

    def __init__(self, csv_path: str, transform=None, **kwargs):
        super().__init__(csv_path, transform)
        self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        self.fns = []
        self.volume_range = {}
        self.ids_to_indices = {}
        for _, row in df.iterrows():
            image_name1, image_name2, image_name3, mask_name = row
            id = osp.splitext(image_name1)[0]  # FLARE22_Tr_0001_0000_0009
            sid = int(id.split("_")[-1])
            pid = "_".join(id.split("_")[:-1])
            self.fns.append(
                {
                    "pid": pid, 
                    "image1": image_name1, 
                    "image2": image_name2, 
                    "image3": image_name3, 
                    "label": mask_name, 
                    "sid": sid
                }
            )
            if pid not in self.volume_range.keys():
                self.volume_range[pid] = []
            self.volume_range[pid].append(sid)
            self.ids_to_indices[id] = len(self.fns) - 1

    def _load_mask(self, idx):
        patient_item = self.fns[idx]
        label_path = patient_item["label"]
        mask = np.load(label_path)  # (H,W) with each pixel value represent one class
        return mask
        
    def __getitem__(self, idx):
        item = self.load_image_and_mask(idx)
        sid = self.fns[idx]['sid']
        pid = self.fns[idx]['pid']
        volume_length = len(self.volume_range[pid])

        return {
            "input": item['image'],
            "target": item['mask'],
            "sid": sid/volume_length, 
            "pid": item["pid"],
            "ori_size": item["ori_size"],
            "img_name": item['img_name'],
        }

    def collate_fn(self, batch):
        imgs = torch.stack([i["input"] for i in batch])
        masks = torch.stack([i["target"] for i in batch])
        pids = [i["pid"] for i in batch]
        sids = [i["sid"] for i in batch]
        ori_sizes = [i["ori_size"] for i in batch]
        img_names = [i["img_name"] for i in batch]
        
        masks = self._encode_masks(masks)
        return {
            "inputs": imgs,
            "targets": masks,
            "pids": pids,
            "sids": sids,
            "ori_sizes": ori_sizes,
            "img_names": img_names,
        }

class FLARE22V2UnlabelledCSVPosDataset(FLARE22V2BaseCSVDataset):
    r"""CSVDataset multi-labels segmentation dataset

    image_dir: `str`
        path to directory contains images
    mask_dir: `str`
        path to directory contains masks
    transform: Optional[List]
        transformatin functions
        
    """

    def __init__(self, root_dir: str, csv_path: str, transform=None, **kwargs):
        super().__init__(root_dir, csv_path, transform)
        self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        self.fns = []
        self.volume_range = {}
        self.ids_to_indices = {}
        for _, row in df.iterrows():
            image_name1, image_name2, image_name3 = row
            id = osp.splitext(image_name1)[0]  # FLARE22_Tr_0001_0000_0009
            sid = int(id.split("_")[-1])
            pid = "_".join(id.split("_")[:-1])
            self.fns.append(
                {
                    "pid": pid, 
                    "image1": image_name1, 
                    "image2": image_name2, 
                    "image3": image_name3, 
                    "sid": sid
                }
            )
            if pid not in self.volume_range.keys():
                self.volume_range[pid] = []
            self.volume_range[pid].append(sid)
            self.ids_to_indices[id] = len(self.fns) - 1

    def _load_mask(self, idx):
        return None

    def __getitem__(self, idx):
        item = self.load_image_and_mask(idx)
        sid = self.fns[idx]['sid']
        pid = self.fns[idx]['pid']
        volume_length = len(self.volume_range[pid])

        return {
            "input": item['image'],
            "pid": item["pid"],
            "sid": sid/volume_length, 
            "ori_size": item["ori_size"],
            "img_name": item['img_name'],
        }

    def collate_fn(self, batch):
        imgs = torch.stack([i["input"] for i in batch])
        pids = [i["pid"] for i in batch]
        sids = [i["sid"] for i in batch]

        ori_sizes = [i["ori_size"] for i in batch]
        img_names = [i["img_name"] for i in batch]

        return {
            "inputs": imgs,
            "pids": pids,
            "sids": sids,
            "ori_sizes": ori_sizes,
            "img_names": img_names,
        }