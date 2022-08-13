from typing import Union, Optional, List, Dict
import os
import os.path as osp
import torch
import numpy as np
from theseus.semantic2D.datasets.flare2022v2.base import (
    FLARE22V2BaseCSVDataset, all_to_onehot
)
import cv2
import pandas as pd
import random
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")

from theseus.semantic2D.utilities.referencer import Referencer
REFERENCER = Referencer()


class FLARE22V2STCNTrainDataset(FLARE22V2BaseCSVDataset):
    """
    Generate pseudo VOS data by applying random transforms on static images.

    FLARE22
    ├── TrainImage
    │   └── <file1>.npy 
    │   ├── <file2>
    ....
    ├── TrainMask
    │   └── <file1>.npy

    volume_dir: `str`
        path to `TrainImage`
    label_dir: `str`
        path to `TrainMask`
    max_jump: `int`
        max number of frames to jump
    """

    def __init__(
        self, root_dir: str, csv_path: str, max_jump: int = 25, use_aug:bool=False, transform=None, **kwargs
    ):

        super().__init__(root_dir, csv_path, transform)
        self.max_jump = max_jump
        self.use_aug = use_aug
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
                    "sid": sid}
            )
            if pid not in self.volume_range.keys():
                self.volume_range[pid] = []
            self.volume_range[pid].append(sid)
            self.ids_to_indices[id] = len(self.fns) - 1

    def random_augment(self, images, masks):
        if random.random() < 0.3:
            images = [torch.flip(i, (1,)) for i in images]
            masks = [torch.flip(i, (0,)) for i in masks]
        
        if random.random() < 0.3:
            images = [torch.flip(i, (2,)) for i in images]
            masks = [torch.flip(i, (1,)) for i in masks]

        # Only degrade the first mask
        # if random.random() < 0.3:
        #     first_mask = masks[0].clone().numpy()
        #     kernel = np.ones((5,5), np.uint8)
        #     first_mask = cv2.erode(first_mask, kernel, iterations=1)
        #     masks[0] = torch.from_numpy(first_mask)

        # elif random.random() < 0.3:
        #     first_mask = masks[0].clone().numpy()
        #     kernel = np.ones((5,5), np.uint8)
        #     first_mask = cv2.dilate(first_mask, kernel, iterations=1)
        #     masks[0] = torch.from_numpy(first_mask)

        return images, masks

    def _load_mask(self, idx):
        patient_item = self.fns[idx]
        label_path = osp.join(self.root_dir, patient_item["label"])
        mask = np.load(label_path)  # (H,W) with each pixel value represent one class
        return mask

    def wrap_item(self, images, masks):
        c, h, w = images[0].shape
        images = torch.stack(images, 0)
        labels = np.unique(masks[0])
        masks = np.stack(masks, 0)

        # Remove background
        labels = labels[labels != 0]

        if len(labels) == 0:
            target_object = -1  # all black if no objects
            has_second_object = False
        else:
            target_object = np.random.choice(labels)
            has_second_object = len(labels) > 1
            if has_second_object:
                labels = labels[labels != target_object]
                second_object = np.random.choice(labels)

        tar_masks = (masks == target_object).astype(np.float32)[:, np.newaxis, :, :]
        if has_second_object:
            sec_masks = (masks == second_object).astype(np.float32)[:, np.newaxis, :, :]
            selector = torch.FloatTensor([1, 1])
        else:
            sec_masks = np.zeros_like(tar_masks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((3, h, w), dtype=np.int)
        cls_gt[tar_masks[:, 0] > 0.5] = 1
        cls_gt[sec_masks[:, 0] > 0.5] = 2

        return images, tar_masks, sec_masks, cls_gt, selector

    def sampling_near_frames(self, item):

        pid = item["pid"]
        current_sid = item["sid"]

        sid_range = self.volume_range[pid]
        min_sid_range = min(sid_range)
        max_sid_range = max(sid_range)

        # Don't want to bias towards beginning/end
        this_max_jump = max(min_sid_range, min(max_sid_range, self.max_jump))
        f1_sid = current_sid + np.random.randint(this_max_jump + 1) + 1
        f1_sid = max(
            min_sid_range,
            min(f1_sid, max_sid_range - this_max_jump, max_sid_range - 1),
        )

        f2_sid = f1_sid + np.random.randint(this_max_jump + 1) + 1
        f2_sid = max(
            min_sid_range,
            min(f2_sid, max_sid_range - this_max_jump // 2, max_sid_range - 1),
        )

        frames_sid = [current_sid, f1_sid, f2_sid]

        if np.random.rand() < 0.5:
            # Reverse time
            frames_sid = frames_sid[::-1]

        images = []
        masks = []
        ids = []
        for f_sid in frames_sid:
            id = pid + "_" + str(f_sid).zfill(4)
            f_idx = self.ids_to_indices[id]
            item = self.load_image_and_mask(f_idx)
            images.append(item["image"].squeeze())
            masks.append(item["mask"].squeeze())
            ids.append(f_idx)

        return images, masks, ids

    def __getitem__(self, idx):
        item = self.fns[idx]
        pid = item["pid"]
        images, masks, frames_idx = self.sampling_near_frames(item)

        if self.use_aug:
            images, masks = self.random_augment(images,masks)

        images, tar_masks, sec_masks, cls_gt, selector = self.wrap_item(images, masks)

        data = {
            "input": images,  # normalized image, torch.Tensor (T, C, H, W)
            "target": tar_masks,  # target mask, numpy (T, 1, H, W) , values 1 at primary class
            "cls_gt": cls_gt,  # numpy (T, H, W), each pixel present one class
            "sec_gt": sec_masks,  # second object mask, numpy (T, 1, H, W) , values 1 at second class
            "selector": selector,  # [1, 1] if has second object, else [1, 0]
            "info": {"name": pid, "frame_indices": frames_idx},
        }
        return data

    def collate_fn(self, batch):

        inputs = torch.stack([item["input"] for item in batch], dim=0)
        targets = torch.stack(
            [torch.from_numpy(item["target"]) for item in batch], dim=0
        )
        cls_gts = torch.stack(
            [torch.from_numpy(item["cls_gt"]) for item in batch], dim=0
        )
        sec_gts = torch.stack(
            [torch.from_numpy(item["sec_gt"]) for item in batch], dim=0
        )
        selectors = torch.stack([item["selector"] for item in batch], dim=0)
        infos = [item["info"] for item in batch]

        return {
            "inputs": inputs,
            "targets": targets,
            "cls_gt": cls_gts,
            "sec_gt": sec_gts,
            "selector": selectors,
            "info": infos,
        }

class FLARE22V2STCNValDataset(FLARE22V2BaseCSVDataset):
    def __init__(self, root_dir: str, csv_path: str, transform=None, **kwargs):
        super().__init__(root_dir, csv_path, transform)
        self.single_object = False
        self._load_data()
        self.compute_stats()

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        self.fns = []
        for _, row in df.iterrows():
            vol_name, mask_name = row
            id = osp.splitext(vol_name)[0]  # FLARE22_Tr_0001_0000
            pid = "_".join(id.split("_")[:-1])
            self.fns.append(
                {
                    "pid": pid, 
                    "vol": vol_name, 
                    "label": mask_name, 
                }
            )

    def compute_stats(self):
        """
        Compute statistic for dataset
        """
        LOGGER.text("Computing statistic...", level=LoggerObserver.INFO)
        self.stats = []
        for item in self.fns:
            vol_dict = {
                "guides": [],
            }
            gt_path = osp.join(self.root_dir, item["label"])
            gt_vol = np.load(gt_path)  # (H, W, NS)
            vol_dict["guides"] = REFERENCER.search_reference(
                gt_vol, strategy="largest-area"
            )
            self.stats.append(vol_dict)

    def _load_item(self, patient_item):
        """
        Load volume with Monai transform
        """
        vol_path = osp.join(self.root_dir, patient_item["vol"])
        npy_image = np.load(vol_path)
        gt_path = osp.join(self.root_dir, patient_item["label"])
        npy_mask = np.load(gt_path)
        out_dict = {
            "image": torch.from_numpy(npy_image),
            "label": torch.from_numpy(npy_mask),
        }
        return out_dict

    def __getitem__(self, idx):
        patient_item = self.fns[idx]
        patient_id = patient_item["pid"]
        item_dict = self._load_item(
            patient_item
        )  # torch.Size([C, H, W, T]), torch.Size([1, H, W, T])
        images = item_dict["image"]
        
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        else:
            images = images / 255.0
            images = images.float()

        ori_vol = item_dict["label"]

        # Choose a reference frame
        stat = self.stats[idx]
        guide_indices = stat["guides"]
        num_slices = images.shape[0]
        masks = []

        # Same for ground truth
        gt_vol = ori_vol.squeeze().numpy()

        # Generate reference frame, only contains first annotation mask
        for f in range(num_slices):
            if f in guide_indices:
                masks.append(gt_vol[f, :, :])
            else:
                masks.append(np.zeros_like(gt_vol[0, :, :]))

        masks = np.stack(masks, 0)

        if self.single_object:
            labels = [1]
            masks = (masks > 0.5).astype(np.uint8)
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()
        else:
            labels = np.array([i for i in range(self.num_classes)])
            labels = labels[labels != 0]
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        masks = masks.unsqueeze(2)
        
        data = {
            "inputs": images,  # (num_slices, C, H, W)
            "gt": masks,  # (C, num_slices, 1, H, W)
            "targets": ori_vol.squeeze()
            .permute(1, 2, 0)
            .numpy(),  # for evaluation (1, H, W, num_slices)
            "info": {  # infos are used for validation and inference
                "name": patient_id,
                "labels": labels,
                "guide_indices": guide_indices,  #
            },
        }

        return data