import argparse
import os
import os.path as osp
import random

import numpy as np
import pandas as pd
from source.semantic3D.augmentations.monai_tf import (
    Compose,
    IntensityClip,
    NormalizeIntensityd,
)
from source.semantic3D.utilities.preprocess.loading import (
    change_axes_of_image,
    load_ct_info,
)
from tqdm import tqdm

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser("Process volume CT")
parser.add_argument("-i", "--input_dir", type=str, help="Volume directory")
parser.add_argument("-o", "--out_dir", type=str, help="Output directory")
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
    vol_path, pbar, gt_path=None, normalize=False,
):
    image_dict = load_ct_info(vol_path)

    if gt_path:
        mask_dict = load_ct_info(gt_path)

    raw_spacing = image_dict["spacing"]
    image_direction = image_dict["direction"]
    subdirection = image_dict["subdirection"]
    origin = image_dict["origin"]

    image_dict["npy_image"] = change_axes_of_image(
        image_dict["npy_image"], subdirection
    )
    if gt_path:
        mask_dict["npy_image"] = change_axes_of_image(
            mask_dict["npy_image"], subdirection
        )

    npy_image = image_dict["npy_image"]
    npy_mask = None

    pbar.set_postfix_str(
        f"Convert {vol_path} from {image_dict['npy_image'].shape} to {npy_image.shape}"
    )

    if gt_path:
        npy_mask = mask_dict["npy_image"]
        pbar.set_postfix_str(
            f"Convert {gt_path} from {mask_dict['npy_image'].shape} to {npy_mask.shape}"
        )
    if normalize:
        if gt_path:
            out_dict = TRANSFORM({"image": npy_image, "label": npy_mask})
            npy_image, npy_mask = out_dict["image"], out_dict["label"]
        else:
            out_dict = TRANSFORM({"image": npy_image,})
            npy_image, npy_mask = out_dict["image"], None

    return {
        "image": npy_image,
        "mask": npy_mask,
    }


def save_npy_volume_mask(
    root_dir,
    save_image_dir,
    save_mask_dir,
    fileid,
    npy_volume,
    npy_mask,
    return_filenames: bool = False,
    non_negative_only: bool = False,
):
    # npy_volume dim: [T, H, W]
    filenames = []
    masknames = []

    for i, (vol, mask) in enumerate(zip(npy_volume, npy_mask)):
        tmp_name = osp.join(fileid, fileid + f"_{str(i).zfill(4)}.npy")
        if non_negative_only and np.sum(mask) == 0:
            continue
        filename = osp.join(root_dir, save_image_dir, tmp_name)
        os.makedirs(osp.join(root_dir, save_image_dir, fileid), exist_ok=True)

        maskname = osp.join(root_dir, save_mask_dir, tmp_name)
        os.makedirs(osp.join(root_dir, save_mask_dir, fileid), exist_ok=True)

        np.save(maskname, mask)
        np.save(filename, vol)

        filenames.append(osp.join(save_image_dir, tmp_name))
        masknames.append(osp.join(save_mask_dir, tmp_name))

    if return_filenames:
        return filenames, masknames


def random_split(filenames, ratio):
    train_filenames = np.random.choice(
        filenames, size=int(ratio * len(filenames)), replace=False
    )
    train_masknames = [i.replace("_0000.nii.gz", ".nii.gz") for i in train_filenames]
    val_filenames = [i for i in filenames if i not in train_filenames]
    val_masknames = [i.replace("_0000.nii.gz", ".nii.gz") for i in val_filenames]

    return train_filenames, train_masknames, val_filenames, val_masknames

def fixed_split(filenames, ratio):

    val_filenames = [
        "FLARE22_Tr_0009_0000.nii.gz",
        "FLARE22_Tr_0012_0000.nii.gz",
        "FLARE22_Tr_0013_0000.nii.gz",
        "FLARE22_Tr_0025_0000.nii.gz",
        "FLARE22_Tr_0035_0000.nii.gz"
    ]
    train_filenames = [i for i in filenames if i not in val_filenames]
    train_masknames = [i.replace("_0000.nii.gz", ".nii.gz") for i in train_filenames]
    val_masknames = [i.replace("_0000.nii.gz", ".nii.gz") for i in val_filenames]

    return train_filenames, train_masknames, val_filenames, val_masknames


def split_train_val(root_dir, out_dir, ratio=0.9):
    filenames = os.listdir(osp.join(root_dir, "images"))
    train_filenames, train_masknames, val_filenames, val_masknames = fixed_split(filenames, ratio)

    target_imagesTr = osp.join(out_dir, "TrainImage")
    target_labelsTr = osp.join(out_dir, "TrainMask")
    os.makedirs(target_imagesTr, exist_ok=True)
    os.makedirs(target_labelsTr, exist_ok=True)

    target_imagesVl = osp.join(out_dir, "ValImage")
    target_labelsVl = osp.join(out_dir, "ValMask")
    os.makedirs(target_imagesVl, exist_ok=True)
    os.makedirs(target_labelsVl, exist_ok=True)

    df_dict = {"train": {"image": [], "label": []}, "val": {"image": [], "label": []}}

    print("Processing train files")
    pbar = tqdm(zip(train_filenames, train_masknames), total=len(train_filenames))
    tbar = tqdm(bar_format="{desc}{postfix}")
    for train_filename, train_maskname in pbar:
        image_path = osp.join(root_dir, "images", train_filename)
        gt_path = osp.join(root_dir, "labels", train_maskname)
        image_dict = convert_2_npy(
            image_path, pbar=tbar, gt_path=gt_path, normalize=False,
        )
        if image_dict["mask"] is not None:
            train_fileid = train_filename.split(".nii.gz")[0]
            imagenames, masknames = save_npy_volume_mask(
                root_dir=out_dir,
                save_image_dir="TrainImage",
                save_mask_dir="TrainMask",
                fileid=train_fileid,
                npy_volume=image_dict["image"].astype(np.float32),
                npy_mask=image_dict["mask"].astype(np.uint8),
                return_filenames=True,
            )
            df_dict["train"]["label"] += masknames
            df_dict["train"]["image"] += imagenames

    pd.DataFrame(df_dict["train"]).to_csv(osp.join(out_dir, "train.csv"), index=False)

    print("Processing val files")
    pbar = tqdm(zip(val_filenames, val_masknames), total=len(val_filenames))
    tbar = tqdm(bar_format="{desc}{postfix}")
    for val_filename, val_maskname in pbar:
        image_path = osp.join(root_dir, "images", val_filename)
        gt_path = osp.join(root_dir, "labels", val_maskname)
        image_dict = convert_2_npy(
            image_path, pbar=tbar, gt_path=gt_path, normalize=False,
        )
        if image_dict["mask"] is not None:
            val_fileid = val_filename.split(".nii.gz")[0]
            imagenames, masknames = save_npy_volume_mask(
                root_dir=out_dir,
                save_image_dir="ValImage",
                save_mask_dir="ValMask",
                fileid=val_fileid,
                npy_volume=image_dict["image"].astype(np.float32),
                npy_mask=image_dict["mask"].astype(np.uint8),
                return_filenames=True,
            )

            df_dict["val"]["label"] += masknames
            df_dict["val"]["image"] += imagenames

    pd.DataFrame(df_dict["val"]).to_csv(osp.join(out_dir, "val.csv"), index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    split_train_val(args.input_dir, args.out_dir, args.ratio)