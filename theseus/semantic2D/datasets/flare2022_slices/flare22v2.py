from typing import Union, Optional, List, Dict
import os
import os.path as osp
import torch
import numpy as np
import pandas as pd
from theseus.semantic2D.datasets.flare2022_slices import (
    FLARE22SlicesBaseDataset,
)
import cv2
import random
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")
class FLARE22SlicesBaseCSVDatasetV2(FLARE22SlicesBaseDataset):
    """
    Load in csv
    """

    def __init__(self, root_dir: str, csv_path: str, transform=None, **kwargs):

        super().__init__(root_dir, transform)
        self.csv_path = csv_path

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

    def load_image_and_mask(self, idx):
        patient_item = self.fns[idx]
        img_path1 = osp.join(self.root_dir, patient_item["image1"])
        img_path2 = osp.join(self.root_dir, patient_item["image2"])
        img_path3 = osp.join(self.root_dir, patient_item["image3"])
        label_path = osp.join(self.root_dir, patient_item["label"])
        img1 = cv2.imread(img_path1, 0)
        img2 = cv2.imread(img_path2, 0)
        img3 = cv2.imread(img_path3, 0)

        img = np.stack([img1, img2, img3], axis=-1)
        img = img / 255.0
        img = img.astype(np.float32)

        width, height = img1.shape
        mask = self._load_mask(label_path)

        if self.transform is not None:
            item = self.transform(image=img, mask=mask)
            img, mask = item["image"], item["mask"]
        
        return {
            "image": img,
            "mask": mask,
            "pid": patient_item["pid"],
            "ori_size": [width, height],
            "img_name": osp.basename(img_path1),
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


class FLARE22SlicesDatasetV2(FLARE22SlicesBaseCSVDatasetV2):
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
        self, root_dir: str, csv_path: str, max_jump: int = 25, use_aug:bool=True, transform=None, **kwargs
    ):

        super().__init__(root_dir, csv_path, transform)
        self.max_jump = max_jump
        self.use_aug = use_aug
        self._load_data()

    def random_augment(self, images, masks):
        if random.random() < 0.3:
            images = [torch.flip(i, (1,)) for i in images]
            masks = [torch.flip(i, (0,)) for i in masks]
        
        if random.random() < 0.3:
            images = [torch.flip(i, (2,)) for i in images]
            masks = [torch.flip(i, (1,)) for i in masks]

        # Only degrade the first mask
        if random.random() < 0.3:
            first_mask = masks[0].clone().numpy()
            kernel = np.ones((5,5), np.uint8)
            first_mask = cv2.erode(first_mask, kernel, iterations=1)
            masks[0] = torch.from_numpy(first_mask)

        elif random.random() < 0.3:
            first_mask = masks[0].clone().numpy()
            kernel = np.ones((5,5), np.uint8)
            first_mask = cv2.dilate(first_mask, kernel, iterations=1)
            masks[0] = torch.from_numpy(first_mask)

        return images, masks

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


# SIMPLE SEGMENTATION DATASET
class FLARE22SlicesNormalDatasetV2(FLARE22SlicesBaseCSVDatasetV2):
    r"""CSVDataset multi-labels segmentation dataset

    Reads in .csv file with structure below:
        filename   | label
        ---------- | -----------
        <img1>.jpg | <mask1>.jpg

    image_dir: `str`
        path to directory contains images
    mask_dir: `str`
        path to directory contains masks
    transform: Optional[List]
        transformatin functions
        
    """

    def __init__(
        self, root_dir: str, csv_path: str, transform: Optional[List] = None, **kwargs
    ):
        super().__init__(root_dir, csv_path, transform)
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.transform = transform
        self._load_data()

    def __getitem__(self, idx):
        item = self.load_image_and_mask(idx)
        return {
            "input": item['image'],
            "target": item['mask'],
            "pid": item["pid"],
            "ori_size": item["ori_size"],
            "img_name": item['img_name'],
        }

    def collate_fn(self, batch):
        imgs = torch.stack([i["input"] for i in batch])
        masks = torch.stack([i["target"] for i in batch])
        pids = [i["pid"] for i in batch]
        ori_sizes = [i["ori_size"] for i in batch]
        img_names = [i["img_name"] for i in batch]
        
        masks = self._encode_masks(masks)
        return {
            "inputs": imgs,
            "targets": masks,
            "pids": pids,
            "ori_sizes": ori_sizes,
            "img_names": img_names,
        }


# SIMPLE SEGMENTATION DATASET
class FLARE22SlicesFolderDatasetV2(FLARE22SlicesBaseDataset):
    r"""CSVDataset multi-labels segmentation dataset

    image_dir: `str`
        path to directory contains images
    mask_dir: `str`
        path to directory contains masks
    transform: Optional[List]
        transformatin functions
        
    """

    def __init__(self, root_dir: str, csv_path: str, transform=None, **kwargs):
        super().__init__(root_dir, transform)
        self.csv_path = csv_path

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
                    "sid": sid}
            )
            if pid not in self.volume_range.keys():
                self.volume_range[pid] = []
            self.volume_range[pid].append(sid)
            self.ids_to_indices[id] = len(self.fns) - 1

    def load_image_and_mask(self, idx):
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

        if self.transform is not None:
            item = self.transform(image=img)
            img = item["image"]
        
        return {
            "image": img,
            "pid": patient_item["pid"],
            "ori_size": [width, height],
            "img_name": osp.basename(img_path1),
        }

    def __getitem__(self, idx):
        item = self.load_image_and_mask(idx)
        return {
            "input": item['image'],
            "pid": item["pid"],
            "ori_size": item["ori_size"],
            "img_name": item['img_name'],
        }

    def collate_fn(self, batch):
        imgs = torch.stack([i["input"] for i in batch])
        pids = [i["pid"] for i in batch]
        ori_sizes = [i["ori_size"] for i in batch]
        img_names = [i["img_name"] for i in batch]

        return {
            "inputs": imgs,
            "pids": pids,
            "ori_sizes": ori_sizes,
            "img_names": img_names,
        }