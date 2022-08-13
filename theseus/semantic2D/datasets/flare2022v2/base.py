from typing import Dict
import os
import os.path as osp
import torch
import numpy as np
import cv2
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
        raise NotImplementedError

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

    def __len__(self) -> int:
        return len(self.fns)


class FLARE22V2BaseDataset(SemanticDataset):
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
        super().__init__()
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

    def load_image_and_mask(self, idx):
        patient_item = self.fns[idx]
        img, width, height, img_name = self._load_image(idx)
        mask = self._load_mask(idx)

        if self.transform is not None:
            if mask is not None:
                item = self.transform(image=img, mask=mask)
                img, mask = item["image"], item["mask"]
            else:
                item = self.transform(image=img)
                img, mask = item["image"], None
        
        return {
            "image": img,
            "mask": mask,
            "pid": patient_item["pid"],
            "ori_size": [width, height],
            "img_name": img_name,
        }

    def _load_image(self, idx):
        patient_item = self.fns[idx]
        img_path1 = osp.join(self.root_dir, patient_item["image1"])
        img_path2 = osp.join(self.root_dir, patient_item["image2"])
        img_path3 = osp.join(self.root_dir, patient_item["image3"])
        img1 = cv2.imread(img_path1, 0)
        img2 = cv2.imread(img_path2, 0)
        img3 = cv2.imread(img_path3, 0)

        img = np.stack([img1, img2, img3], axis=-1)
        img = img / 255.0
        img = img.astype(np.float32)

        width, height = img1.shape
        return img, width, height, osp.basename(img_path1)

    def _load_mask(self, idx):
        raise NotImplementedError

    # def _calculate_classes_dist(self):
    #     classes_dist = []    
    #     LOGGER.text("Calculating class frequency...", level=LoggerObserver.DEBUG)
    #     for idx in range(len(self.fns)):
    #         mask = self._load_mask(idx)

    #         classes_dist.append()

class FLARE22V2BaseCSVDataset(FLARE22V2BaseDataset):
    """
    Load in csv
    """

    def __init__(self, root_dir: str, csv_path: str, transform=None, **kwargs):
        super().__init__(root_dir, transform)
        self.csv_path = csv_path

