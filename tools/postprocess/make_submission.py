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
from theseus.semantic3D.utilities.preprocess.loading import save_ct_from_npy, load_ct_info, change_axes_of_image

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser("PostProcess volume CT, resize to original size for submission")
parser.add_argument("-p", "--pred_dir", type=str, help="Volume directory contains prediction images")
parser.add_argument("-g", "--gt_dir", type=str, help="Volume directory contains raw images")
parser.add_argument("-o", "--out_dir", type=str, help="Output directory")

NUM_LABELS = 5

def convert_2_npy(vol_path, target_size=(160,160,160)):
    image_dict = load_ct_info(vol_path)
    raw_spacing = image_dict['spacing']
    image_direction = image_dict['direction']
    origin = image_dict['origin']
    npy_mask, _ = ScipyResample.resample_mask_to_size(
        image_dict['npy_image'], target_size, num_label=NUM_LABELS
    )
    print(f"Convert {vol_path} from {image_dict['npy_image'].shape} to {npy_mask.shape}")
    return {
        'mask': npy_mask,
        'spacing': raw_spacing,
        'direction': image_direction,
        'origin': origin,
    }

def postprocess(pred_dir, gt_dir, out_dir):
    filenames = os.listdir(gt_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("Processing prediction files")
    for test_filename in tqdm(filenames):
        raw_image_path = osp.join(gt_dir, test_filename)

        pred_image_path = osp.join(pred_dir, test_filename)
        assert osp.isfile(pred_image_path), f"Missing {pred_image_path}"

        raw_image_dict = load_ct_info(raw_image_path)
        pred_image_dict = convert_2_npy(pred_image_path, target_size=raw_image_dict['npy_image'].shape)
        pred_image_dict['mask'] = change_axes_of_image(pred_image_dict['mask'], raw_image_dict['subdirection'])

        test_filename = test_filename.split('.')[0] + '.nii.gz'
        dest_image_path = osp.join(out_dir, test_filename)


        save_ct_from_npy(
            npy_image=pred_image_dict['mask'],
            save_path=dest_image_path,
            origin=raw_image_dict['origin'],
            spacing=raw_image_dict['spacing'],
            direction=raw_image_dict['direction'],
            sitk_type=sitk.sitkUInt8
        )
    

if __name__ == '__main__':
    args = parser.parse_args()
    postprocess(args.pred_dir, args.gt_dir, args.out_dir)