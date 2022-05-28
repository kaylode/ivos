from typing import Union
import torch
import numpy as np
import os
from theseus.semantic2D.utilities.sampling import sampling_frames
from theseus.semantic2D.datasets.flare2022 import FLARE22BaseDataset
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class FLARE22TestDataset(FLARE22BaseDataset):
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
        self.load_data()

    def load_data(self):
        volume_names = sorted(os.listdir(self.root_dir))
        self.fns = []
        for volume_name in volume_names:
            # train_0047_0000.nii.gz
            pid = volume_name
            self.fns.append(
                {"pid": pid, "vol": volume_name,}
            )

    def __getitem__(self, idx):
        patient_item = self.fns[idx]
        patient_id = patient_item["pid"]
        item_dict = self._load_item(patient_item)
        image = item_dict["image"]  # torch.Size([H, W, T])
        affine = item_dict["affine"]
        case_spacing = item_dict["spacing"]
        width, height, num_slices = image.shape

        # Full volume # (C, H, W, NS) --> (NS, C, H, W)
        full_images = image.permute(2, 0, 1).unsqueeze(1)

        # Reference frames
        images = []
        frames_idx = sampling_frames(
            num_slices,
            max_frames=self.max_ref_frames,
            uniform=True,
            sampling_rate=self.sample_fp,
        )

        for f_idx in frames_idx:
            this_im = image[:, :, f_idx]  # (H, W)
            images.append(this_im)
        images = torch.stack(images, 0).unsqueeze(1)

        return {
            "ref_image": images,
            "ref_indices": frames_idx,
            "full_image": full_images,
            "info": {
                "img_name": patient_id,
                "ori_size": [width, height],
                "affine": affine,
                "spacing": case_spacing,
            },
        }

    def collate_fn(self, batch):
        imgs = torch.cat([i["ref_image"] for i in batch], dim=0)
        full_images = torch.stack([i["full_image"] for i in batch], dim=0)
        infos = [i["info"] for i in batch]
        ref_indices = [i["ref_indices"] for i in batch]

        return {
            "ref_images": imgs,
            "full_images": full_images,
            "ref_indices": ref_indices,
            "infos": infos,
        }
