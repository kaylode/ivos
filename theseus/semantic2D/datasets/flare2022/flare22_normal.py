import os.path as osp
import nibabel as nib # common way of importing nibabel
import torch
import numpy as np
import pandas as pd
from theseus.semantic2D.utilities.sampling import sampling_frames
from theseus.semantic2D.datasets.flare2022 import FLARE22BaseCSVDataset
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger('main')

# CLASS FOR SIMPLE SEGMENTATION MODELS

class FLARE22NormalDataset(FLARE22BaseCSVDataset):
    def __init__(
        self, 
        root_dir: str,
        csv_path: str,
        max_frames: int,
        transform=None,
        **kwargs):

        super().__init__(root_dir, csv_path, transform)
        self.max_frames = max_frames

    def sampling_frames(self, num_frames):
        return sampling_frames(num_frames, max_frames=self.max_frames)

    def __getitem__(self, idx):
        patient_item = self.fns[idx]
        patient_id = patient_item['pid']
        item_dict = self._load_item(patient_item) 
        image = item_dict['image']              # torch.Size([H, W, T])
        gt_vol = item_dict['label'].squeeze(0)  #  torch.Size([H, W, T])
        affine = item_dict['affine']
        case_spacing = item_dict['spacing']

        width, height, num_slices = image.shape
        frames_idx = self.sampling_frames(num_slices)
       
        images = []
        masks = []
        for f_idx in frames_idx:
            this_im = image[:,:,f_idx] #(H, W)
            this_gt = gt_vol[:,:,f_idx] #(H, W)

            images.append(this_im)
            masks.append(this_gt)
        
        images = torch.stack(images, 0).unsqueeze(1)
        masks = torch.stack(masks, 0)

        
        return {
            "input": images, 
            'target': masks,
            'img_name': patient_id,
            'ori_size': [width, height]
        }

    def _encode_masks(self, masks):
        """
        Input masks from _load_mask(), but in shape [B, H, W]
        Output should be one-hot encoding of segmentation masks [B, NC, H, W]
        """
        
        one_hot = torch.nn.functional.one_hot(masks.long(), num_classes=self.num_classes) # (B,H,W,NC)
        one_hot = one_hot.permute(0, 3, 1, 2) # (B,NC,H,W)
        return one_hot.float()

    def collate_fn(self, batch):
        imgs = torch.cat([i['input'] for i in batch], dim=0)
        masks = torch.cat([i['target'] for i in batch], dim=0)
        img_names = [i['img_name'] for i in batch]
        ori_sizes = [i['ori_size'] for i in batch]
        
        masks = self._encode_masks(masks)
        return {
            'inputs': imgs,
            'targets': masks,
            'img_names': img_names,
            'ori_sizes': ori_sizes
        }

class FLARE22NormalValDataset(FLARE22NormalDataset):

    def __init__(self, root_dir, csv_path, sample_fp=0, max_frames=-1, transform=None):
        super().__init__(root_dir, csv_path, max_frames, transform)
        self.sample_fp = sample_fp

    def sampling_frames(self, num_frames):
        return sampling_frames(
            num_frames, 
            max_frames=self.max_frames, 
            uniform=True, 
            sampling_rate=self.sample_fp)
