from typing import Iterable, List
import os
import os.path as osp
from PIL import Image
import torch
import torch.utils.data as data
import numpy as np
import nibabel as nib
from theseus.semantic2D.utilities.sampling import sampling_frames

class VolumeFolderDataset(data.Dataset):
    """
    Dataset contains folder of volume 

    root_dir: `str`
        path to folder of images

    sample_fp: `int`
        steps between samples
    
    max_frames: `int`
        maximum number of frames
    """
    def __init__(self, root_dir, sample_fp, max_frames, transform: List =None, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.sample_fp = sample_fp
        self.max_frames = max_frames
        self.transform = transform
        self.load_data()

    def load_data(self):
        """
        Load filepaths into memory
        """
        self.fns = []
        image_names = os.listdir(self.root_dir)
        for image_name in image_names:
            self.fns.append(image_name)

    def _load_item(self, vol_path):
        """
        Load volume with Monai transform
        """
        nib_label = nib.load(vol_path)
        affine = nib_label.affine
        case_spacing = nib_label.header.get_zooms()
        out_dict = self.transform({
            'image': [vol_path],
            'affine': affine, 
            'spacing':  case_spacing
        })
        return out_dict

    def __getitem__(self, index: int):
        """
        Get an item from memory
        """
        image_name = self.fns[index]
        image_path = os.path.join(self.root_dir, image_name)
        item_dict = self._load_item(image_path)
        image = item_dict['image']

        width, height, num_slices = image.shape
        frames_idx = sampling_frames(
            num_slices, 
            max_frames=self.max_frames, 
            uniform=True, 
            sampling_rate=self.sample_fp)
       
        images = []
        for f_idx in frames_idx:
            this_im = image[:,:,f_idx] #(H, W)
            images.append(this_im)
        
        images = torch.stack(images, 0).unsqueeze(1)
        
        return {
            "input": images, 
            "img_name": image_name,
            'ori_size': [width, height, num_slices]
        }

    def __len__(self):
        return len(self.fns)

    def collate_fn(self, batch):
        imgs = torch.cat([i['input'] for i in batch], dim=0)
        img_names = [i['img_name'] for i in batch]
        ori_sizes = [i['ori_size'] for i in batch]
        
        masks = self._encode_masks(masks)
        return {
            'inputs': imgs,
            'img_names': img_names,
            'ori_sizes': ori_sizes
        }