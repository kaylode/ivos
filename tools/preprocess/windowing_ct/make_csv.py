import os
import os.path as osp
import pandas as pd
from tqdm import tqdm

NPY_IMG_TRAIN="data/flare22/npy/training/TrainImage"
NPY_MSK_TRAIN="data/flare22/npy/training/TrainMask"
NPY_IMG_VAL="data/flare22/npy/training/ValImage"
NPY_MSK_VAL="data/flare22/npy/training/ValMask"
NPY_IMG_VALIDATION="data/flare22/npy/validation/ValidationImage"
NPY_MSK_VALIDATION="data/flare22/npy/validation/ValidationMask"
NPY_CSV_TRAIN="data/flare22/npy/train_npy.csv"
NPY_CSV_VAL="data/flare22/npy/val_npy.csv"
NPY_CSV_VALIDATION="data/flare22/npy/validation_npy.csv"

SLICES_IMG_TRAIN="data/flare22/slices/training/TrainImage"
SLICES_MSK_TRAIN="data/flare22/slices/training/TrainMask"
SLICES_IMG_VAL="data/flare22/slices/training/ValImage"
SLICES_MSK_VAL="data/flare22/slices/training/ValMask"
SLICES_IMG_VALIDATION="data/flare22/slices/validation/ValidationImage"
SLICES_MSK_VALIDATION="data/flare22/slices/validation/ValidationMask"
SLICES_CSV_TRAIN="data/flare22/slices/train_slices.csv"
SLICES_CSV_VAL="data/flare22/slices/val_slices.csv"
SLICES_CSV_VALIDATION="data/flare22/slices/validation_slices.csv"

TRAIN_FORMAT= "FLARE22_Tr_{pid}_0000_{sid}.jpg"
VALIDATION_FORMAT= "FLARETs_{pid}_0000_{sid}.jpg"

def run_slices_train(img_dir, gt_dir, out_csv, name_format):

    if gt_dir is not None:
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

    filenames = os.listdir(img_dir)
    for filename in tqdm(filenames):

        if name_format == TRAIN_FORMAT:
            idx = filename.split('_')[2]
        else:
            idx = filename.split('_')[1]

        folder1 = f"abdomen-soft tissues_abdomen-liver"
        folder2 = f"chest-lungs_chest-mediastinum"
        folder3 = f"spine-bone"
        tmp_path = osp.join(img_dir, filename, folder1)
        
        sids = len(os.listdir(tmp_path))
        mid_range = [0, sids]

        for sid in range(mid_range[0], mid_range[1]):
            image_name = name_format.format(
                pid=str(idx).zfill(4),
                sid=str(sid).zfill(4)
            )
            image_path1 = osp.join(img_dir, filename, folder1, image_name)
            image_path2 = osp.join(img_dir, filename, folder2, image_name)
            image_path3 = osp.join(img_dir, filename, folder3, image_name)
            df['image1'].append(image_path1)
            df['image2'].append(image_path2)
            df['image3'].append(image_path3)
            if gt_dir is not None:
                label_path = osp.join(gt_dir, filename, image_name[:-4]+'.npy')
                df['label'].append(label_path)

    pd.DataFrame(df).to_csv(out_csv, index=False)

def run_npy_train(img_dir, gt_dir, out_csv):
    if gt_dir is not None:
        df = {
            'image': [],
            'label': []
        }
    else:
        df = {
            'image': [],
        }

    filenames = os.listdir(img_dir)
    for filename in tqdm(filenames):
        image_path = osp.join(img_dir, filename)
        df['image'].append(image_path)
        if gt_dir is not None:
            label_path = osp.join(gt_dir, filename[:-4]+'.npy')
            df['label'].append(label_path)

    pd.DataFrame(df).to_csv(out_csv, index=False)


if __name__ == '__main__':
    run_slices_train(
        SLICES_IMG_TRAIN, 
        SLICES_MSK_TRAIN, 
        SLICES_CSV_TRAIN,
        name_format=TRAIN_FORMAT)

    run_slices_train(
        SLICES_IMG_VAL, 
        SLICES_MSK_VAL, 
        SLICES_CSV_VAL,
        name_format=TRAIN_FORMAT)

    run_slices_train(
        SLICES_IMG_VALIDATION, 
        SLICES_MSK_VALIDATION, 
        SLICES_CSV_VALIDATION,
        name_format=VALIDATION_FORMAT)

    run_npy_train(
        NPY_IMG_TRAIN, 
        NPY_MSK_TRAIN, 
        NPY_CSV_TRAIN
    )

    run_npy_train(
        NPY_IMG_VAL, 
        NPY_MSK_VAL, 
        NPY_CSV_VAL
    )

    run_npy_train(
        NPY_IMG_VALIDATION, 
        NPY_MSK_VALIDATION, 
        NPY_CSV_VALIDATION
    )
