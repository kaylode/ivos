import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_dir', help='Input path to image')
parser.add_argument('-g', '--gt_dir', default=None, help='Input path to label dir')
parser.add_argument('-s', '--save_dir', help='Save path')
parser.add_argument('-o', '--output_csv', help='Output path to save csv file')

def run(args):
    os.makedirs(args.save_dir, exist_ok=True)
    volname_ls = os.listdir(args.image_dir)
    for volname in tqdm(volname_ls):
        filename = f"{volname}/abdomen-soft tissues_abdomen-liver"
        filename2 = f"{volname}/chest-lungs_chest-mediastinum"
        filename3 = f"{volname}/spine-bone"

        foldername = osp.join(args.image_dir, filename)
        foldername2 = osp.join(args.image_dir, filename2)
        foldername3 = osp.join(args.image_dir, filename3)

        sids = sorted(os.listdir(foldername))

        vol = []
        for sid in sids:
            sidname = sid

            filepath = osp.join(foldername, sidname)
            filepath2 = osp.join(foldername2, sidname)
            filepath3 = osp.join(foldername3, sidname)

            im = cv2.imread(filepath, 0)
            im2 = cv2.imread(filepath2, 0)
            im3 = cv2.imread(filepath3, 0)
            
            stacked = np.stack([im, im2, im3], axis=0)
            vol.append(stacked)

        vol = np.stack(vol, axis=0)
        np.save(osp.join(args.save_dir, f"{volname}.npy"), vol)

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)