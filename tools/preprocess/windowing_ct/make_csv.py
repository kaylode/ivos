import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_dir', help='Input path to image')
parser.add_argument('-g', '--gt_dir', default=None, help='Input path to label dir')
parser.add_argument('-t', '--type', default='slices', help='Process slices or npy')
parser.add_argument('-o', '--output_csv', help='Output path to save csv file')


def run_slices(args):

    if args.gt_dir is not None:
        df = {
            'image1': [],
            'image2': [],
            'image3': [],
            'label': []
        }
    else:
        df = {
            'image1': [],
            'image2': [],
            'image3': [],
        }

    filenames = os.listdir(args.image_dir)
    for filename in tqdm(filenames):
        idx = filename.split('_')[2]
        folder1 = f"abdomen-soft tissues_abdomen-liver"
        folder2 = f"chest-lungs_chest-mediastinum"
        folder3 = f"spine-bone"
        tmp_path = osp.join(args.image_dir, filename, folder1)
        
        sids = len(os.listdir(tmp_path))
        mid_range = [0, sids]

        for sid in range(mid_range[0], mid_range[1]):
            image_name = f"FLARE22_Tr_{str(idx).zfill(4)}_0000_{str(sid).zfill(4)}.jpg"
            image_path1 = osp.join(args.image_dir, filename, folder1, image_name)
            image_path2 = osp.join(args.image_dir, filename, folder2, image_name)
            image_path3 = osp.join(args.image_dir, filename, folder3, image_name)
            df['image1'].append(image_path1)
            df['image2'].append(image_path2)
            df['image3'].append(image_path3)
            if args.gt_dir is not None:
                label_path = osp.join(args.gt_dir, filename, image_name[:-4]+'.npy')
                df['label'].append(label_path)

    pd.DataFrame(df).to_csv(args.output_csv, index=False)

def run_npy(args):
    if args.gt_dir is not None:
        df = {
            'image': [],
            'label': []
        }
    else:
        df = {
            'image': [],
        }

    filenames = os.listdir(args.image_dir)
    for filename in tqdm(filenames):
        image_name = f"FLARE22_Tr_{str(idx).zfill(4)}_0000_{str(sid).zfill(4)}.jpg"
        image_path1 = osp.join(args.image_dir, filename, folder1, image_name)
        df['image1'].append(image_path1)
        if args.gt_dir is not None:
            label_path = osp.join(args.gt_dir, filename, image_name[:-4]+'.npy')
            df['label'].append(label_path)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
