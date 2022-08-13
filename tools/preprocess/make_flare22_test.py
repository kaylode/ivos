import argparse
import os
import os.path as osp
import random

import numpy as np
import pandas as pd
import SimpleITK as sitk
from theseus.semantic3D.augmentations.monai_tf import (
    Compose,
    IntensityClip,
    NormalizeIntensityd,
)
from theseus.semantic3D.utilities.preprocess.loading import (
    change_axes_of_image,
    load_ct_info,
    save_ct_from_npy,
)
from theseus.semantic3D.utilities.preprocess.resampler import ItkResample, ScipyResample
from tqdm import tqdm

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser("Process volume CT")
parser.add_argument("-i", "--input_dir", type=str, help="Volume directory")
parser.add_argument("-o", "--out_dir", type=str, help="Output directory")

"""
flare22
    ├── Validation
    │   └── <file1>.nii.gz 
    │   ├── <file2>
    ....
"""

NUM_LABELS = 14
TARGET_TRAIN_SIZE = [-1, 512, 512]
TARGET_TEST_SIZE = [-1, 512, 512]
TRANSFORM = Compose(
    [
        # PercentileClip(keys=['image'],min_pct=2, max_pct=98), or
        IntensityClip(keys=["image"], min_value=-3024.0, max_value=3024.0),
        NormalizeIntensityd(keys=["image"]),
    ]
)

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


def convert_2_npy(
    vol_path,
    pbar,
    gt_path=None,
    target_size=(160, 160, 160),
    normalize=True,
):
    image_dict = load_ct_info(vol_path)

    if target_size[0] == -1:
        image_shape = image_dict["npy_image"].shape
        target_size[0] = image_shape[0]

    if gt_path:
        mask_dict = load_ct_info(gt_path)

    raw_spacing = image_dict["spacing"]
    image_direction = image_dict["direction"]
    subdirection = image_dict["subdirection"]
    origin = image_dict["origin"]

    image_dict["npy_image"] = change_axes_of_image(
        image_dict["npy_image"], subdirection
    )

    npy_image, zoom_factor = ScipyResample.resample_to_size(
        image_dict["npy_image"], target_size
    )


    if normalize:
        if gt_path:
            out_dict = TRANSFORM({"image": npy_image, "label": npy_mask})
            npy_image, npy_mask = out_dict["image"], out_dict["label"]
        else:
            out_dict = TRANSFORM(
                {
                    "image": npy_image,
                }
            )
            npy_image, npy_mask = out_dict["image"], None
            
    pbar.set_postfix_str(
        f"Convert {vol_path} from {image_dict['npy_image'].shape} to {npy_image.shape}"
    )

    return {
        "image": npy_image,
        "mask": npy_mask,
        "spacing": raw_spacing,
        "direction": image_direction,
        "origin": origin,
        "zoom_factor": zoom_factor,
    }


def split_train_val(root_dir, out_dir):
    assert osp.exists(root_dir)
    # test_filenames = os.listdir(osp.join(root_dir, "Validation"))
    # target_imagesTs = osp.join(out_dir, "Validation")
    test_filenames = os.listdir(root_dir)
    target_imagesTs = osp.join(out_dir)
    os.makedirs(target_imagesTs, exist_ok=True)

    print("Processing test files")
    pbar = tqdm(test_filenames, total=len(test_filenames))
    tbar = tqdm(bar_format="{desc}{postfix}")
    for test_filename in pbar:
        image_path = osp.join(root_dir, test_filename)
        image_dict = convert_2_npy(
            image_path,
            pbar=tbar,
            gt_path=None,
            target_size=TARGET_TEST_SIZE[:],
            normalize=True,
        )

        dest_image_path = osp.join(target_imagesTs, test_filename)

        save_ct_from_npy(
            npy_image=image_dict["image"],
            save_path=dest_image_path,
            origin=image_dict["origin"],
            spacing=image_dict["spacing"],
            direction=image_dict["direction"],
            sitk_type=sitk.sitkFloat32,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    split_train_val(args.input_dir, args.out_dir)
