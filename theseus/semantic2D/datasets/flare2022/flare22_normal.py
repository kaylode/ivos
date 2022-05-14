import os.path as osp
import nibabel as nib # common way of importing nibabel
import torch
import numpy as np
import pandas as pd
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
        if self.max_frames == -1:
            self.max_frames = num_frames
            
        frames_idx = np.random.choice(range(num_frames), size=self.max_frames, replace=False)
        return frames_idx

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
        """sample video into tensor
        Args:
            video_file: location of video file
            max_frame: max frame number
            sample_fp: sampling rate
        Returns:
            image_input: sample frames
        """

        assert self.sample_fp > -1

        if self.max_frames == -1:
            self.max_frames = num_frames

        # Pre-uniform sampling
        current_frame = num_frames // self.sample_fp # number of frames based on rate
        current_sample_indx = np.linspace(
          0, num_frames - 1, 
          num=current_frame, dtype=int) # sample frames, with step equals sample_fp
        
        # if the length of current_sample_indx is already less than max_frames, just use the current version to tensor
        # else continue to uniformly sample the frames whose length is max_frames
        # when training, the frames are sampled randomly in the uniform split interval
        if self.max_frames >=  current_sample_indx.shape[0]:
            frame_index = np.arange(0, current_sample_indx.shape[0])
        else:
            frame_index = np.linspace(0, current_sample_indx.shape[0] - 1, num=self.max_frames, dtype=int)

        sampled_frame_ids = [current_sample_indx[int(index)] for index in frame_index]
        
        return sampled_frame_ids
