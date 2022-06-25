import argparse
import os
import os.path as osp
import random

import numpy as np
import pandas as pd
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
TARGET_TRAIN_SIZE = [512, 512]

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


def cut_into_slices(root_dir, out_dir):
    assert osp.exists(root_dir)
    test_filenames = os.listdir(root_dir)
    os.makedirs(out_dir, exist_ok=True)

    df_dict = {
        'image1': [],
        'image2': [],
        'image3': [],
        'label': [],
    }

    print("Processing test files")
    pbar = tqdm(test_filenames, total=len(test_filenames))
    tbar = tqdm(bar_format="{desc}{postfix}")
    for test_filename in pbar:
        image_path = osp.join(root_dir, test_filename)
        test_filename = test_filename.split('.')[0]
        image = np.load(image_path)
        weight, height, depth = image.shape

        for i in range(depth):
            frame = image[..., i]
            foldername = osp.join(out_dir, test_filename)
            os.makedirs(foldername, exist_ok=True)
            outpath = osp.join(foldername, f"{test_filename}_{str(i).zfill(4)}.npy") 
            np.save(outpath, frame)      

if __name__ == "__main__":
    args = parser.parse_args()
    cut_into_slices(args.input_dir, args.out_dir)
