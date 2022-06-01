from typing import Union
import os.path as osp
import nibabel as nib  # common way of importing nibabel
import torch
import numpy as np
from theseus.semantic2D.datasets.flare2022 import FLARE22BaseCSVDataset, all_to_onehot
from theseus.utilities.loggers.observer import LoggerObserver
from theseus.semantic2D.utilities.sampling import sampling_frames
from theseus.semantic2D.utilities.referencer import Referencer

LOGGER = LoggerObserver.getLogger("main")


REFERENCER = Referencer()


class FLARE22TrainDataset(FLARE22BaseCSVDataset):
    """
    Generate pseudo VOS data by applying random transforms on static images.
    Single-object only.

    FLARE22
    ├── TrainImage
    │   └── <file1>.nii.gz 
    │   ├── <file2>
    ....
    ├── TrainMask
    │   └── <file1>.nii.gz

    volume_dir: `str`
        path to `TrainImage`
    label_dir: `str`
        path to `TrainMask`
    max_jump: `int`
        max number of frames to jump
    """

    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        max_jump: int = 25,
        max_frames: Union[int, float] = 5,
        transform=None,
        **kwargs
    ):

        super().__init__(root_dir, csv_path, transform)
        self.max_jump = max_jump
        self.max_frames = max_frames
        self.compute_stats()

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
            gt_vol = nib.load(gt_path).get_fdata()  # (H, W, NS)
            gt_vol = gt_vol.transpose(2, 0, 1)
            vol_dict["guides"] = REFERENCER.search_reference(
                gt_vol, strategy="non-empty"
            )
            self.stats.append(vol_dict)

    def sampling_frames(self, num_frames):
        return sampling_frames(num_frames, max_frames=self.max_frames)

    def wrap_item(self, images, masks):

        h, w = images[0].shape
        images = torch.stack(images, 0).unsqueeze(1)
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

    def __getitem__(self, idx):

        patient_item = self.fns[idx]
        patient_id = patient_item["pid"]
        item_dict = self._load_item(
            patient_item
        )  # torch.Size([C, H, W, T]), torch.Size([1, H, W, T])

        image = item_dict["image"]
        gt_vol = item_dict["label"]
        affine = item_dict["affine"]
        case_spacing = item_dict["spacing"]

        gt_vol = gt_vol.squeeze(0)
        h, w, num_slices = image.shape

        stat = self.stats[idx]
        sampling_indices = self.sampling_frames(len(stat["guides"]))
        candidate_indices = [stat["guides"][i] for i in sampling_indices]

        batch = []
        for start_idx in candidate_indices:
            # Don't want to bias towards beginning/end
            this_max_jump = min(num_slices, self.max_jump)
            f1_idx = start_idx + np.random.randint(this_max_jump + 1) + 1
            f1_idx = min(f1_idx, num_slices - this_max_jump, num_slices - 1)

            f2_idx = f1_idx + np.random.randint(this_max_jump + 1) + 1
            f2_idx = min(f2_idx, num_slices - this_max_jump // 2, num_slices - 1)

            frames_idx = [start_idx, f1_idx, f2_idx]

            if np.random.rand() < 0.5 and f2_idx in stat["guides"]:
                # Reverse time
                frames_idx = frames_idx[::-1]

            images = []
            masks = []
            for f_idx in frames_idx:
                this_im = image[:, :, f_idx]  # (H, W)
                this_gt = gt_vol[:, :, f_idx]  # (H, W)
                this_gt = this_gt.numpy()
                images.append(this_im)
                masks.append(this_gt)

            images, tar_masks, sec_masks, cls_gt, selector = self.wrap_item(
                images, masks
            )

            data = {
                "input": images,  # normalized image, torch.Tensor (T, C, H, W)
                "target": tar_masks,  # target mask, numpy (T, 1, H, W) , values 1 at primary class
                "cls_gt": cls_gt,  # numpy (T, H, W), each pixel present one class
                "sec_gt": sec_masks,  # second object mask, numpy (T, 1, H, W) , values 1 at second class
                "selector": selector,  # [1, 1] if has second object, else [1, 0]
                "info": {
                    "name": patient_id,
                    "slice_id": frames_idx,
                    "affine": affine,
                    "case_spacing": case_spacing,
                },
            }

            batch.append(data)

        return batch

    def __len__(self):
        return len(self.fns)

    def collate_fn(self, batch):

        new_batch = [item for sublist in batch for item in sublist]

        inputs = torch.stack([item["input"] for item in new_batch], dim=0)
        targets = torch.stack(
            [torch.from_numpy(item["target"]) for item in new_batch], dim=0
        )
        cls_gts = torch.stack(
            [torch.from_numpy(item["cls_gt"]) for item in new_batch], dim=0
        )
        sec_gts = torch.stack(
            [torch.from_numpy(item["sec_gt"]) for item in new_batch], dim=0
        )
        selectors = torch.stack([item["selector"] for item in new_batch], dim=0)
        infos = [item["info"] for item in new_batch]

        return {
            "inputs": inputs,
            "targets": targets,
            "cls_gt": cls_gts,
            "sec_gt": sec_gts,
            "selector": selectors,
            "info": infos,
        }


class FLARE22ValDataset(FLARE22BaseCSVDataset):
    def __init__(self, root_dir: str, csv_path: str, transform=None, **kwargs):

        super().__init__(root_dir, csv_path, transform)
        self.single_object = False
        self.compute_stats()

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
            gt_vol = nib.load(gt_path).get_fdata()  # (H, W, NS)
            gt_vol = gt_vol.transpose(2, 0, 1)
            vol_dict["guides"] = REFERENCER.search_reference(
                gt_vol, strategy="largest-area"
            )
            self.stats.append(vol_dict)

    def __getitem__(self, idx):
        patient_item = self.fns[idx]
        patient_id = patient_item["pid"]
        item_dict = self._load_item(
            patient_item
        )  # torch.Size([C, H, W, T]), torch.Size([1, H, W, T])
        image = item_dict["image"]
        ori_vol = item_dict["label"]
        affine = item_dict["affine"]
        case_spacing = item_dict["spacing"]
        images = image.permute(2, 0, 1).unsqueeze(1)  # (C, H, W, NS) --> (NS, C, H, W)

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
                masks.append(gt_vol[:, :, f])
            else:
                masks.append(np.zeros_like(gt_vol[:, :, 0]))

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
            "targets": ori_vol.squeeze().numpy(),  # for evaluation (1, H, W, num_slices)
            "info": {  # infos are used for validation and inference
                "name": patient_id,
                "labels": labels,
                "guide_indices": guide_indices,  #
                "affine": affine,  # from nib.load
                "case_spacing": case_spacing,
            },
        }

        return data


class FLARE22NumpyValDataset(FLARE22BaseCSVDataset):
    def __init__(self, root_dir: str, csv_path: str, transform=None, **kwargs):

        super().__init__(root_dir, csv_path, transform)
        self.single_object = False
        self.compute_stats()

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

        if "label" in patient_item.keys():
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
        images = item_dict["image"].unsqueeze(1)
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
