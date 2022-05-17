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
from theseus.semantic3D.augmentations.monai_tf import (
    PercentileClip,
    IntensityClip,
    NormalizeIntensityd,
    Compose
)
from theseus.semantic3D.utilities.preprocess.resampler import ItkResample, ScipyResample
from theseus.semantic3D.utilities.preprocess.loading import save_ct_from_npy, load_ct_info, change_axes_of_image

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser("Process volume CT")
parser.add_argument("-i", "--input_dir", type=str, help="Volume directory")
parser.add_argument("-o", "--out_dir", type=str, help="Output directory")
parser.add_argument("--binary", action='store_true',  help="Whether to split into binary masks")
parser.add_argument("--ratio", type=float, default=0.9, help="Ratio split")

"""
flare22
    ├── TrainImage
    │   └── <file1>.nii.gz 
    │   ├── <file2>
    ....
    ├── TrainMask
    │   └── <file1>.nii.gz
    ....
    ├── Validation
    │   └── <file1>.nii.gz 
    │   ├── <file2>
    ....
"""

NUM_LABELS = 14
TARGET_TRAIN_SIZE = [320,320,320]
TARGET_TEST_SIZE = [512,512,512]
TRANSFORM = Compose([
    # PercentileClip(keys=['image'],min_pct=2, max_pct=98), or
    IntensityClip(keys=['image'], min_value=-3024.0, max_value=3024.0),
    NormalizeIntensityd(keys=['image'])
])

CLASSNAMES = [
    "background",
    "liver", 
    "kidney_r",
    "spleen",
    "pancreas",
    "aorta",
    "IVC",
    "RAG",
    "LAG",
    "gallbladder",
    "esophagus",
    "stomach",
    "duodenum",
    "kidney_l",
]

def convert_2_npy(vol_path, gt_path=None, target_size=(160,160,160), normalize=True, binary=False):
    image_dict = load_ct_info(vol_path)

    if target_size[0] == -1:
        image_shape = image_dict['npy_image'].shape
        target_size[0] = image_shape[0]

    if gt_path:
        mask_dict = load_ct_info(gt_path)

    raw_spacing = image_dict['spacing']
    image_direction = image_dict['direction']
    subdirection = image_dict['subdirection']
    origin = image_dict['origin']

    image_dict['npy_image'] = change_axes_of_image(image_dict['npy_image'], subdirection)
    if gt_path:
        mask_dict['npy_image'] = change_axes_of_image(mask_dict['npy_image'], subdirection)

    npy_image, zoom_factor = ScipyResample.resample_to_size(image_dict['npy_image'], target_size)
    
    print(f"Convert {vol_path} from {image_dict['npy_image'].shape} to {npy_image.shape}")

    if gt_path:
        npy_mask, _ = ScipyResample.resample_mask_to_size(
            mask_dict['npy_image'], target_size, num_label=NUM_LABELS
        )
        print(f"Convert {gt_path} from {mask_dict['npy_image'].shape} to {npy_mask.shape}")

    if normalize:
        if gt_path:
            out_dict = TRANSFORM({
                'image': npy_image,
                'label': npy_mask
            })
            npy_image, npy_mask = out_dict['image'], out_dict['label']
        else:
            out_dict = TRANSFORM({
                'image': npy_image,
            })
            npy_image, npy_mask = out_dict['image'], None

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
    test_filenames = os.listdir(osp.join(root_dir, 'Validation'))

    train_filenames = np.random.choice(filenames, size=int(ratio*len(filenames)), replace=False)
    train_masknames = ['_'.join(i.split('_')[:-1])+'.nii.gz' for i in train_filenames]

    val_filenames = [i for i in filenames if i not in train_filenames]
    val_masknames = ['_'.join(i.split('_')[:-1])+'.nii.gz' for i in val_filenames]

    target_imagesTr = osp.join(out_dir, "TrainImage")
    target_labelsTr = osp.join(out_dir, "TrainMask")
    os.makedirs(target_imagesTr, exist_ok=True)
    os.makedirs(target_labelsTr, exist_ok=True)

    target_imagesVl = osp.join(out_dir, "ValImage")
    target_labelsVl = osp.join(out_dir, "ValMask")
    os.makedirs(target_imagesVl, exist_ok=True)
    os.makedirs(target_labelsVl, exist_ok=True)

    target_imagesTs = osp.join(out_dir, "Validation")
    os.makedirs(target_imagesTs, exist_ok=True)

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

    print("Processing train files")
    for train_filename, train_maskname in tqdm(zip(train_filenames, train_masknames)):
        image_path = osp.join(root_dir, 'TrainImage', train_filename)
        gt_path = osp.join(root_dir, 'TrainMask', train_maskname)
        image_dict = convert_2_npy(image_path, gt_path, target_size=TARGET_TRAIN_SIZE[:], normalize=True)
        
        dest_image_path = osp.join(target_imagesTr, train_filename)


        save_ct_from_npy(
            npy_image=image_dict['image'],
            save_path=dest_image_path,
            origin=image_dict['origin'],
            spacing=image_dict['spacing'],
            direction=image_dict['direction'],
            sitk_type=sitk.sitkFloat32
        )

        if image_dict['mask'] is not None:
            if args.binary:
                mask = image_dict['mask']
                labels = np.unique(mask)
                for label in labels:
                    if label == 0:
                        continue
                    tmp_mask = mask == label
                    os.makedirs(osp.join(target_labelsTr, CLASSNAMES[label]), exist_ok=True)
                    dest_gt_path = osp.join(target_labelsTr, CLASSNAMES[label], train_maskname)
                    df_dict['train']['image'].append(osp.join("TrainImage", train_filename))
                    df_dict['train']['label'].append(osp.join("TrainMask", CLASSNAMES[label], train_maskname))
                    save_ct_from_npy(
                        npy_image=tmp_mask,
                        save_path=dest_gt_path,
                        origin=image_dict['origin'],
                        spacing=image_dict['spacing'],
                        direction=image_dict['direction'],
                        sitk_type=sitk.sitkUInt8
                    )
            else:
                df_dict['train']['image'].append(osp.join("TrainImage", train_filename))
                df_dict['train']['label'].append(osp.join("TrainMask", train_maskname))
                dest_gt_path = osp.join(target_labelsTr, train_maskname)
                save_ct_from_npy(
                    npy_image=image_dict['mask'],
                    save_path=dest_gt_path,
                    origin=image_dict['origin'],
                    spacing=image_dict['spacing'],
                    direction=image_dict['direction'],
                    sitk_type=sitk.sitkUInt8
                )

    print("Processing val files")
    for val_filename, val_maskname in tqdm(zip(val_filenames, val_masknames)):
        image_path = osp.join(root_dir, 'TrainImage', val_filename)
        gt_path = osp.join(root_dir, 'TrainMask', val_maskname)

        image_dict = convert_2_npy(image_path, gt_path, target_size=TARGET_TRAIN_SIZE[:], normalize=True)

        dest_image_path = osp.join(target_imagesVl, val_filename)
        dest_gt_path = osp.join(target_labelsVl, val_maskname)

        df_dict['val']['image'].append(osp.join("ValImage", val_filename))
        df_dict['val']['label'].append(osp.join("ValMask", val_maskname))

        save_ct_from_npy(
            npy_image=image_dict['image'],
            save_path=dest_image_path,
            origin=image_dict['origin'],
            spacing=image_dict['spacing'],
            direction=image_dict['direction'],
            sitk_type=sitk.sitkFloat32
        )

        if image_dict['mask'] is not None:
            save_ct_from_npy(
                npy_image=image_dict['mask'],
                save_path=dest_gt_path,
                origin=image_dict['origin'],
                spacing=image_dict['spacing'],
                direction=image_dict['direction'],
                sitk_type=sitk.sitkUInt8
            )

    pd.DataFrame(df_dict['train']).to_csv(osp.join(out_dir, 'train.csv'), index=False)
    pd.DataFrame(df_dict['val']).to_csv(osp.join(out_dir, 'val.csv'), index=False)
    
    print("Processing test files")
    for test_filename in tqdm(test_filenames):
        image_path = osp.join(root_dir, 'Validation', test_filename)
        image_dict = convert_2_npy(image_path, gt_path=None, target_size=TARGET_TEST_SIZE[:], normalize=True)

        dest_image_path = osp.join(target_imagesTs, test_filename)

        save_ct_from_npy(
            npy_image=image_dict['image'],
            save_path=dest_image_path,
            origin=image_dict['origin'],
            spacing=image_dict['spacing'],
            direction=image_dict['direction'],
            sitk_type=sitk.sitkFloat32
        )
    

if __name__ == '__main__':
    args = parser.parse_args()
    split_train_val(args.input_dir, args.out_dir, args.ratio)