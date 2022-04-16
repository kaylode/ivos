import shutil
import os
import os.path as osp
import numpy as np
import argparse
import random

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser("Process volume CT")
parser.add_argument("-i", "--input_dir", type=str, help="Volume directory")
parser.add_argument("-o", "--out_dir", type=str, help="Output directory")
parser.add_argument("--ratio", type=float, default=0.9, help="Ratio split")


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

    for train_filename, train_maskname in zip(train_filenames, train_masknames):
        src_path = osp.join(root_dir, 'TrainImage', train_filename)
        dest_path = osp.join(target_imagesTr, train_filename)
        shutil.copy(src_path, dest_path)

        src_path = osp.join(root_dir, 'TrainMask', train_maskname)
        dest_path = osp.join(target_labelsTr, train_maskname)
        shutil.copy(src_path, dest_path)

    for val_filename, val_maskname in zip(val_filenames, val_masknames):
        src_path = osp.join(root_dir, 'TrainImage', val_filename)
        dest_path = osp.join(target_imagesVl, val_filename)
        shutil.copy(src_path, dest_path)

        src_path = osp.join(root_dir, 'TrainMask', val_maskname)
        dest_path = osp.join(target_labelsVl, val_maskname)
        shutil.copy(src_path, dest_path)

if __name__ == '__main__':
    args = parser.parse_args()
    split_train_val(args.input_dir, args.out_dir, args.ratio)