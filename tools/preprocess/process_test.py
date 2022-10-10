import argparse
import os
import os.path as osp
import random
import numpy as np
from tqdm import tqdm

from tools.preprocess.split_train_val import (
    convert_2_npy, save_npy_volume_mask
)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser("Process volume CT")
parser.add_argument("-i", "--input_dir", type=str, help="Volume directory")
parser.add_argument("-l", "--label_dir", type=str, default=None, help="Volume directory")
parser.add_argument("-o", "--out_dir", type=str, help="Output directory")
parser.add_argument("-t", "--type", type=str, help="Folder type")

def save_npy_volume(
    save_image_dir, fileid, npy_volume, return_filenames: bool = False,
):
    # npy_volume dim: [T, H, W]
    filenames = []

    for i, vol in enumerate(npy_volume):
        tmp_name = osp.join(fileid, fileid + f"_{str(i).zfill(4)}.npy")
        filename = osp.join(save_image_dir, tmp_name)
        os.makedirs(osp.join(save_image_dir, fileid), exist_ok=True)
        np.save(filename, vol)
        filenames.append(tmp_name)

    if return_filenames:
        return filenames

def process_test(root_dir, out_dir, tag, label_dir=None):
    print("Processing test files")
    test_filenames = os.listdir(root_dir)
    pbar = tqdm(test_filenames, total=len(test_filenames))
    tbar = tqdm(bar_format="{desc}{postfix}")

    for test_filename in pbar:
        test_fileid = test_filename.split(".nii.gz")[0]
        image_path = osp.join(root_dir, test_filename)

        if label_dir:
            gt_path = osp.join(label_dir, test_filename.replace("_0000.nii.gz", ".nii.gz"))
            if osp.exists(gt_path):
                image_dict = convert_2_npy(image_path, pbar=tbar, gt_path=gt_path, normalize=False)
                save_npy_volume_mask(
                    root_dir=out_dir,
                    save_image_dir=f"{tag}Image",
                    save_mask_dir=f"{tag}Mask",
                    fileid=test_fileid,
                    npy_volume=image_dict["image"].astype(np.float32),
                    npy_mask=image_dict["mask"].astype(np.uint8),
                    return_filenames=False,
                )
                continue
        
        image_dict = convert_2_npy(image_path, pbar=tbar, gt_path=None, normalize=False)
        save_npy_volume(
            save_image_dir=osp.join(out_dir,f"{tag}Image"),
            fileid=test_fileid,
            npy_volume=image_dict["image"].astype(np.float32),
            return_filenames=False,
        )

if __name__ == "__main__":
    args = parser.parse_args()
    process_test(args.input_dir, args.out_dir, tag=args.type, label_dir=args.label_dir)
