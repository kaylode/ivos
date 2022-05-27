from typing import Dict
import os
import os.path as osp
import torch
import numpy as np
import pandas as pd
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros(
            (len(labels), masks.shape[0], masks.shape[1], masks.shape[2]),
            dtype=np.uint8,
        )
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for k, l in enumerate(labels):
        Ms[k] = (masks == l).astype(np.uint8)

    return Ms


class SemanticDataset(torch.utils.data.Dataset):
    r"""Base dataset for segmentation tasks
    """

    def __init__(self, **kwawrgs):
        self.classes_idx = {}
        self.num_classes = 0
        self.classnames = None
        self.transform = None
        self.image_dir = None
        self.mask_dir = None
        self.fns = []

    def _load_data(self):
        raise NotImplementedError

    def collate_fn(self, batch):
        imgs = torch.stack([i["input"] for i in batch])
        masks = torch.stack([i["target"]["mask"] for i in batch])
        img_names = [i["img_name"] for i in batch]
        ori_sizes = [i["ori_size"] for i in batch]

        masks = self._encode_masks(masks)
        return {
            "inputs": imgs,
            "targets": masks,
            "img_names": img_names,
            "ori_sizes": ori_sizes,
        }

    def __len__(self) -> int:
        return len(self.fns)


class FLARE22SlicesBaseDataset(SemanticDataset):
    """
    Dataset to load FLARE22 data.

    FLARE22
    ├── TrainImage
    │   └── <file1>.npy
    │   ├── <file2>
    ....
    ├── TrainMask
    │   └── <file1>.npy

    root_dir: `str`
        path to root_dir
    """

    def __init__(self, root_dir: str, transform=None, **kwargs):

        self.root_dir = root_dir
        self.transform = transform
        self.classnames = [
            "background",
            "liver",
            "kidney_r",
            "spleen",
            "pancreas",
            "aorta",
            "IVC",
            "RAG",
            "LAG",
            "gallbladder",
            "esophagus",
            "stomach",
            "duodenum",
            "kidney_l",
        ]

        self.classes_idx = {}
        self.num_classes = len(self.classnames)


class FLARE22SlicesBaseCSVDataset(FLARE22SlicesBaseDataset):
    """
    Load in csv
    """

    def __init__(self, root_dir: str, csv_path: str, transform=None, **kwargs):

        super().__init__(root_dir, transform)
        self.csv_path = csv_path
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.csv_path)
        self.fns = []
        self.volume_range = {}
        self.ids_to_indices = {}
        for _, row in df.iterrows():
            image_name, mask_name = row
            id = osp.splitext(image_name)[0]  # FLARE22_Tr_0001_0000_0009
            sid = int(id.split("_")[-1])
            pid = "_".join(id.split("_")[:-1])
            self.fns.append(
                {"pid": pid, "image": image_name, "label": mask_name, "sid": sid}
            )
            if pid not in self.volume_range.keys():
                self.volume_range[pid] = []
            self.volume_range[pid].append(sid)
            self.ids_to_indices[id] = len(self.fns) - 1

    def load_image_and_mask(self, idx):
        patient_item = self.fns[idx]
        img_path = osp.join(self.image_dir, patient_item["image"])
        label_path = osp.join(self.mask_dir, patient_item["label"])
        img = np.load(img_path)
        width, height = img.shape
        mask = self._load_mask(label_path)

        if self.transform is not None:
            item = self.transform(image=img, mask=mask)
            img, mask = item["image"], item["mask"]

        return {
            "image": img,
            "mask": mask,
            "pid": patient_item["pid"],
            "ori_size": [width, height],
        }

    def _load_mask(self, label_path):
        mask = np.load(label_path)  # (H,W) with each pixel value represent one class
        return mask

    def _encode_masks(self, masks):
        """
        Input masks from _load_mask(), but in shape [B, H, W]
        Output should be one-hot encoding of segmentation masks [B, NC, H, W]
        """

        one_hot = torch.nn.functional.one_hot(
            masks.long(), num_classes=self.num_classes
        )  # (B,H,W,NC)
        one_hot = one_hot.permute(0, 3, 1, 2)  # (B,NC,H,W)
        return one_hot.float()
