from typing import Union
import torch
import numpy as np
import os
import os.path as osp
from theseus.semantic2D.utilities.sampling import sampling_frames
from theseus.semantic2D.datasets.flare2022v2.base import FLARE22V2BaseDataset
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class FLARE22V2BaseTestDataset(FLARE22V2BaseDataset):
    def __init__(self, root_dir: str, transform=None, **kwargs):
        super().__init__(root_dir, transform, **kwargs)
        self._load_data()

    def _load_data(self):
        volume_names = sorted(os.listdir(self.root_dir))
        self.fns = []
        for volume_name in volume_names:
            # FLARETs_0001_0000.npy
            pid = volume_name.split('.')[0]
            self.fns.append(
                {"pid": pid, "vol": volume_name}
            )

    def _load_image(self, idx):
        """
        Load volume with Monai transform
        """
        patient_item = self.fns[idx]
        vol_path = osp.join(self.root_dir, patient_item['vol'])
        np_vol = np.load(vol_path)
        tensor_vol = torch.from_numpy(np_vol)
        tensor_vol = tensor_vol / 255.0
        tensor_vol = tensor_vol.float()
        return tensor_vol

class FLARE22V2TestDataset(FLARE22V2BaseTestDataset):
    def __init__(
        self,
        root_dir: str,
        sample_fp: int = 5,
        max_ref_frames: Union[int, float] = 15,
        transform=None,
        **kwargs
    ):

        super().__init__(root_dir, transform)
        self.max_ref_frames = max_ref_frames
        self.sample_fp = sample_fp

    def __getitem__(self, idx):
        patient_item = self.fns[idx]
        patient_id = patient_item["pid"]
        full_images = self._load_image(idx) # torch.Size([H, W, T])

        # Full volume  (NS, C, H, W)
        num_slices, c, width, height = full_images.shape

        # Reference frames
        images = []
        frames_idx = sampling_frames(
            num_slices,
            max_frames=self.max_ref_frames,
            uniform=True,
            sampling_rate=self.sample_fp,
        )

        for f_idx in frames_idx:
            this_im = full_images[f_idx,...]  # (H, W)
            images.append(this_im)
        images = torch.stack(images, 0)

        return {
            "ref_image": images,
            "ref_indices": frames_idx,
            "full_image": full_images,
            "info": {
                "img_name": patient_id,
                "ori_size": [width, height],
            },
        }

    def collate_fn(self, batch):
        imgs = torch.cat([i["ref_image"] for i in batch], dim=0)
        full_images = [i["full_image"] for i in batch]
        infos = [i["info"] for i in batch]
        ref_indices = [i["ref_indices"] for i in batch]

        return {
            "ref_images": imgs,
            "full_images": full_images,
            "ref_indices": ref_indices,
            "infos": infos,
        }

class FLARE22V2CoarseMaskTestDataset(FLARE22V2TestDataset):
    def __init__(
        self, 
        root_dir: str,
        mask_dir: str,
        sample_fp: int = 5,
        max_ref_frames: Union[int, float] = 15,
        transform=None,
        ) -> None:
        super().__init__(root_dir, sample_fp, max_ref_frames, transform)
        self.mask_dir = mask_dir
    
    def _load_mask(self, idx):
        """
        Load volume with Monai transform
        """
        patient_item = self.fns[idx]
        vol_path = osp.join(self.mask_dir, patient_item['vol'])
        np_vol = np.load(vol_path).astype(np.uint8)
        tensor_vol = torch.from_numpy(np_vol)

        return tensor_vol

    def __getitem__(self, idx):
        patient_item = self.fns[idx]
        patient_id = patient_item["pid"]
        full_images = self._load_image(idx) # torch.Size([H, W, T])
        full_masks = self._load_mask(idx)

        # Full volume  (NS, C, H, W)
        num_slices, c, width, height = full_images.shape

        # Reference frames
        images = []
        masks = []
        frames_idx = sampling_frames(
            num_slices,
            max_frames=self.max_ref_frames,
            uniform=True,
            sampling_rate=self.sample_fp,
        )

        for f_idx in frames_idx:
            this_im = full_images[f_idx,...]  # (H, W)
            this_mask = full_masks[...,f_idx]  # (H, W)
            images.append(this_im)
            masks.append(this_mask)

        images = torch.stack(images, 0)
        masks = torch.stack(masks, 0)

        return {
            "ref_image": images,
            "ref_mask": masks,
            "ref_indices": frames_idx,
            "full_image": full_images,
            "full_mask": full_masks,
            "info": {
                "img_name": patient_id,
                "ori_size": [width, height],
            },
        }

    def collate_fn(self, batch):
        imgs = torch.cat([i["ref_image"] for i in batch], dim=0)
        masks = torch.cat([i["ref_mask"] for i in batch], dim=0)
        full_images = [i["full_image"] for i in batch]
        full_masks = [i["full_mask"] for i in batch]
        infos = [i["info"] for i in batch]
        ref_indices = [i["ref_indices"] for i in batch]

        return {
            "ref_images": imgs,
            "ref_masks": masks,
            "full_images": full_images,
            "full_masks": full_masks,
            "ref_indices": ref_indices,
            "infos": infos,
        }