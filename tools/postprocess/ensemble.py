import shutil
import os
import os.path as osp
import numpy as np
import argparse
import random
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
from theseus.semantic3D.utilities.preprocess.resampler import ItkResample, ScipyResample
from theseus.semantic3D.utilities.preprocess.loading import (
    save_ct_from_npy,
    load_ct_info,
    change_axes_of_image,
)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser(
    "PostProcess volume CT, resize to original size for submission"
)
parser.add_argument(
    "-p", "--pred_dir", type=str, help="Volume directory contains prediction images"
)
parser.add_argument(
    "-w", "--weight", type=str, help="Path to weight.txt"
)
parser.add_argument("-o", "--out_dir", type=str, help="Output directory")

NUM_LABELS = 14

"""
FLARE22
    ├── run1
    │   └── <file1>.nii.gz 
    │   ├── <file2>.nii.gz 
    ....
    ├── run2
    │   └── <file1>.nii.gz
"""


def convert_2_npy(vol_path):
    image_dict = load_ct_info(vol_path)
    raw_spacing = image_dict["spacing"]
    image_direction = image_dict["direction"]
    origin = image_dict["origin"]

    npy_mask = image_dict["npy_image"]

    print(
        f"Convert {vol_path} from {image_dict['npy_image'].shape} to {npy_mask.shape}"
    )
    return {
        "mask": npy_mask,
        "spacing": raw_spacing,
        "direction": image_direction,
        "origin": origin,
    }

def ensemble(list_of_dicts):
    masks = []
    for i, fdict in enumerate(list_of_dicts):
        mask = fdict["mask"] # (T,H,W)
        if i == 0:
            origin = fdict["origin"]
            spacing = fdict["spacing"]
            direction = fdict["direction"]
        masks.append(mask)

    masks = np.stack(masks, axis=0).transpose(1,0,2,3) # (N,T,H,W)

    N, T, H, W = masks.shape
    results = []
    for mask in masks:
        tmp_mask = mask.reshape(N, H*W)
        


    masks = np.

    return {
        'mask': masks,
        'origin': origin,
        'spacing': spacing,
        'direction': direction
    }

        




def postprocess(pred_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    run_names = os.listdir(pred_dir)
    filenames = os.listdir(osp.join(pred_dir, run_names[0]))

    print("Processing prediction files")
    for test_filename in tqdm(filenames):
        pred_list = []
        for run_name in run_names:
            test_filepath = osp.join(pred_dir, run_name, test_filename)
            pred_image_dict = convert_2_npy(test_filepath)
            pred_list.append(pred_image_dict)
        result = ensemble(pred_list)

        dest_image_path = osp.join(out_dir, test_filename)

        save_ct_from_npy(
            npy_image=pred_image_dict["mask"],
            save_path=dest_image_path,
            origin=raw_image_dict["origin"],
            spacing=raw_image_dict["spacing"],
            direction=raw_image_dict["direction"],
            sitk_type=sitk.sitkUInt8,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    postprocess(args.pred_dir, args.gt_dir, args.out_dir)
