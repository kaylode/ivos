import os
import os.path as osp
import nibabel as nib
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser('Test')
parser.add_argument("-i", "--input_dir", help='Input')
parser.add_argument("-o", "--output_dir", help='Output')

group_dict = {
    1: [10, 11, 12],
    2: [2, 7],
    3: [8, 13],
    4: [1, 9],
    5: [3, 4],
    6: [5, 6],
}

group_mapping = {
    1: 4,
    2: 2,
    3: 5,
    4: 5,
    5: 6,
    6: 6,
    7: 2,
    8: 3,
    9: 4,
    10: 1,
    11: 1,
    12: 1,
    13: 3
}


def split_classes_into_groups(msk):
    new_msk = np.zeros_like(msk)
    for group_id, cls_ids in group_dict.items():
        group_msk = np.isin(msk, cls_ids)
        new_msk[group_msk] = group_id
    return new_msk.astype(np.uint8)

def convert(args):
    os.makedirs(args.output_dir, exist_ok=True)
    filenames = os.listdir(args.input_dir)
    for filename in tqdm(filenames):
        filepath = osp.join(args.input_dir, filename)
        new_path = osp.join(args.output_dir, filename)
        nib_file = nib.load(filepath)
        npy_mask = nib_file.get_fdata()
        new_mask_npy = split_classes_into_groups(npy_mask)
        new_mask = nib.Nifti1Image(new_mask_npy, nib_file.affine)
        nib.save(new_mask, new_path)


if __name__ == '__main__':
    args = parser.parse_args()
    convert(args)