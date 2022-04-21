import shutil
import os
import os.path as osp
import numpy as np
import argparse
import random
import numpy as np
import SimpleITK as sitk
import pandas as pd
from theseus.semantic3D.augmentations.monai_tf import (
    PercentileClip,
    NormalizeIntensityd,
    Compose
)
from theseus.semantic3D.utilities.preprocess.resampler import ItkResample, ScipyResample
from theseus.semantic3D.utilities.preprocess.loading import save_ct_from_npy, load_ct_info

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser("Process volume CT")
parser.add_argument("-i", "--input_dir", type=str, help="Volume directory")
parser.add_argument("-o", "--out_dir", type=str, help="Output directory")
parser.add_argument("--ratio", type=float, default=0.9, help="Ratio split")

"""
abdomenct1k
    ├── TrainImage
    │   └── <file1>.nii.gz 
    │   ├── <file2>
    ....
    ├── TrainMask
    │   └── <file1>.nii.gz
"""

NUM_LABELS = 5
TARGET_SIZE = (160,160,160)
TRANSFORM = Compose([
    PercentileClip(keys=['image'],min_pct=2, max_pct=98),
    NormalizeIntensityd(keys=['image', 'label'])
])

def convert_2_npy(vol_path, gt_path, target_size=(160,160,160), normalize=True):
    image_dict = load_ct_info(vol_path)
    mask_dict = load_ct_info(gt_path)
    npy_image, zoom_factor = ScipyResample.resample_to_size(image_dict['npy_image'], target_size)
    npy_mask, _ = ScipyResample.resample_mask_to_size(
        mask_dict['npy_image'], target_size, num_label=NUM_LABELS
    )

    if normalize:
        out_dict = TRANSFORM({
            'image': npy_image,
            'label': npy_mask
        })
        npy_image, npy_mask = out_dict['image'], out_dict['label']

    raw_spacing = image_dict['spacing']
    image_direction = image_dict['direction']
    origin = image_dict['origin']

    return {
        'image': npy_image,
        'mask': npy_mask,
        'spacing': raw_spacing,
        'direction': image_direction,
        'origin': origin,
        'zoom_factor': zoom_factor
    }

def split_train_val(root_dir, out_dir, ratio=0.9):
    filenames = os.listdir(osp.join(root_dir, 'TrainImage'))

    train_filenames = np.random.choice(filenames, size=int(ratio*len(filenames)), replace=False)
    train_masknames = ['_'.join(i.split('_')[:2])+'.nii.gz' for i in train_filenames]

    val_filenames = [i for i in filenames if i not in train_filenames]
    val_masknames = ['_'.join(i.split('_')[:2])+'.nii.gz' for i in val_filenames]

    target_imagesTr = osp.join(out_dir, "TrainImage")
    target_labelsTr = osp.join(out_dir, "TrainMask")
    os.makedirs(target_imagesTr, exist_ok=True)
    os.makedirs(target_labelsTr, exist_ok=True)

    target_imagesVl = osp.join(out_dir, "ValImage")
    target_labelsVl = osp.join(out_dir, "ValMask")
    os.makedirs(target_imagesVl, exist_ok=True)
    os.makedirs(target_labelsVl, exist_ok=True)

    df_dict = {
        'train': {
            'image': [],
            'label': []
        },
        'val': {
            'image': [],
            'label': []
        }
    }

    for train_filename, train_maskname in zip(train_filenames, train_masknames):
        image_path = osp.join(root_dir, 'TrainImage', train_filename)
        gt_path = osp.join(root_dir, 'TrainMask', train_maskname)
        image_dict = convert_2_npy(image_path, gt_path, target_size=TARGET_SIZE, normalize=True)

        dest_image_path = osp.join(target_imagesTr, train_filename)
        dest_gt_path = osp.join(target_labelsTr, train_maskname)

        df_dict['train']['image'].append(osp.join("TrainImage", train_filename))
        df_dict['train']['label'].append(osp.join("TrainMask", train_maskname))

        save_ct_from_npy(
            npy_image=image_dict['image'],
            save_path=dest_image_path,
            origin=image_dict['origin'],
            spacing=image_dict['spacing'],
            direction=image_dict['direction']
        )

        save_ct_from_npy(
            npy_image=image_dict['mask'],
            save_path=dest_gt_path,
            origin=image_dict['origin'],
            spacing=image_dict['spacing'],
            direction=image_dict['direction']
        )

    for val_filename, val_maskname in zip(val_filenames, val_masknames):
        image_path = osp.join(root_dir, 'TrainImage', val_filename)
        gt_path = osp.join(root_dir, 'TrainMask', val_maskname)

        image_dict = convert_2_npy(image_path, gt_path, target_size=TARGET_SIZE, normalize=True)

        dest_image_path = osp.join(target_imagesVl, val_filename)
        dest_gt_path = osp.join(target_labelsVl, val_maskname)

        df_dict['val']['image'].append(osp.join("ValImage", val_filename))
        df_dict['val']['label'].append(osp.join("ValMask", val_maskname))

        save_ct_from_npy(
            npy_image=image_dict['image'],
            save_path=dest_image_path,
            origin=image_dict['origin'],
            spacing=image_dict['spacing'],
            direction=image_dict['direction']
        )

        save_ct_from_npy(
            npy_image=image_dict['mask'],
            save_path=dest_gt_path,
            origin=image_dict['origin'],
            spacing=image_dict['spacing'],
            direction=image_dict['direction']
        )
    
    pd.DataFrame(df_dict['train']).to_csv(osp.join(out_dir, 'train.csv'), index=False)
    pd.DataFrame(df_dict['val']).to_csv(osp.join(out_dir, 'val.csv'), index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    split_train_val(args.input_dir, args.out_dir, args.ratio)