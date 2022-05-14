import os
import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from theseus.semantic3D.utilities.preprocess.loading import  load_ct_info

parser = argparse.ArgumentParser("Perform data analysis on volume CT")
parser.add_argument("-i", "--image_dir", type=str, help="Volume directory")
parser.add_argument("-l", "--mask_dir", default=None, type=str, help="Volume directory")

def analyze_intensity(image, mask=None):
    if mask is not None:
        binary_mask = mask != 0
        mask_image = binary_mask*image
    else:
        mask_image = np.zeros(image.shape)
    return {
        'min_cls_int' : mask_image.min(), 
        'max_cls_int' : mask_image.max(), 
        'min_int' : image.min(), 
        'max_int' : image.max(), 
    }


def analyze(args):
    filenames = os.listdir(args.image_dir)
    
    depth_list = []
    intensity_list = []
    cls_intensity_list = []

    for filename in tqdm(filenames):
        filepath = osp.join(args.image_dir, filename)
        image_info = load_ct_info(filepath)

        if args.mask_dir:
            filename = filename.replace('_0000.nii.gz', '.nii.gz')
            maskfilepath = osp.join(args.mask_dir, filename)
            mask_info = load_ct_info(maskfilepath)
            intensity_dict = analyze_intensity(image_info['npy_image'], mask_info['npy_image'])
        else:
            intensity_dict = analyze_intensity(image_info['npy_image'], None)

        image_shape =  image_info['npy_image'].shape

        depth_list.append(image_shape[0])
        intensity_list.append(intensity_dict['min_int'])
        intensity_list.append(intensity_dict['max_int'])

        cls_intensity_list.append(intensity_dict['min_cls_int'])
        cls_intensity_list.append(intensity_dict['max_cls_int'])

    info_dict = {
        'min_intensity': [min(intensity_list)],
        'max_intensity': [max(intensity_list)],
        'min_cls_intensity': [min(cls_intensity_list)],
        'max_cls_intensity': [max(cls_intensity_list)],
        'min_depth': [min(depth_list)],
        'max_depth': [max(depth_list)],
    }

    # Print table
    table = tabulate(
        info_dict, headers="keys", tablefmt="fancy_grid"
    )

    print(table)
        


if __name__ == '__main__':
    args = parser.parse_args()
    analyze(args)