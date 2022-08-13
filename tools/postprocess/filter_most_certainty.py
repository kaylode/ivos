from collections import defaultdict
from yaml import parse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
import os.path as osp
import imageio
from tqdm import tqdm
from theseus.utilities.visualization.visualizer import Visualizer
import torch
import argparse

parser = argparse.ArgumentParser(
    "Filter uncertainty"
)
parser.add_argument(
    "-p", "--pred_dir", type=str, help="Volume directory contains prediction numpy images"
)

parser.add_argument(
    "-t", "--threshold", type=float, default=0.9, help="Uncertainty threshold"
)

parser.add_argument(
    "-g", "--pgt_dir", type=str, default=None, help="Volume directory contains best prediction numpy images"
)

parser.add_argument("-o", "--out_dir", type=str, help="Output directory")

NUM_CLASSESS = 14


def compute_certainty_level(pred_masks, target_mask):
    N, W, H, T, C = pred_masks.shape

    miou_scores = []
    for mid in range(N):
        mscore = np.zeros(NUM_CLASSESS) 
        for cl in range(1, NUM_CLASSESS):
            cl_pred = pred_masks[mid,:,:,:,cl] # outputs: (batch, W, H)
            cl_target = target_mask[...,cl] # targets: (batch, W, H)
            score = binary_compute(cl_pred, cl_target)
            mscore[cl] += score
        final_score = sum(mscore) / (NUM_CLASSESS - 1) # subtract background
        miou_scores.append(final_score)
    
    return np.mean(miou_scores)

def binary_compute(predict, target):
    if torch.sum(predict)==0 and torch.sum(target)==0:
        return 1.0
    elif torch.sum(target)==0 and torch.sum(predict)>0:
        return 0.0
    else:
        volume_sum = target.sum() + predict.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (target & predict).sum()
        return volume_intersect / (volume_sum - volume_intersect)

def encode_masks(masks):
    """
    Input masks from _load_mask(), but in shape [B, H, W]
    Output should be one-hot encoding of segmentation masks [B, NC, H, W]
    """
    one_hot = torch.nn.functional.one_hot(
        masks.long(), num_classes=NUM_CLASSESS
    )  # (B,H,W,NC)
    return one_hot.float()

def get_certainty_score_one_mask(list_of_masks, target_mask=None, num_partritions=3):
    N, H, W, T = list_of_masks.shape
    list_of_masks = torch.from_numpy(list_of_masks)

    # Split along time dimension to avoid memory overload
    masks_splits = torch.split(list_of_masks, T//num_partritions, dim=-1)

    if target_mask is not None:
        target_mask = torch.from_numpy(target_mask)
        target_splits = torch.split(target_mask, T//num_partritions, dim=-1)

    total_score = []
    result = []
    for i, m_split in enumerate(masks_splits):
        one_hot_mask = encode_masks(m_split)

        if target_mask is not None:
            ensembled = target_splits[i]
        else:
            one_hot_mask_sum = torch.sum(one_hot_mask, dim=0)
            ensembled = torch.argmax(one_hot_mask_sum, dim=-1)
        one_hot_ensembled = encode_masks(ensembled)
        split_score = compute_certainty_level(one_hot_mask.long(), one_hot_ensembled.long())
        total_score.append(split_score)

        result.append(ensembled)
    result = torch.cat(result, dim=-1)
    return result.numpy().astype(np.uint8), np.mean(total_score)

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Ensemble these runs:")
    run_names = os.listdir(args.pred_dir)
    for run_name in run_names:
        print(run_name)

    num_good = 0
    filenames = os.listdir(osp.join(args.pred_dir, run_name, 'test/masks'))
    for filename in tqdm(filenames):
        masks = []

        if args.pgt_dir is not None:
            npy_path = osp.join(args.pgt_dir, filename)
            if not osp.isfile(npy_path):
                continue
            target_mask = np.load(npy_path)
        else:
            target_mask = None

        for i, run_name in enumerate(run_names):
            npy_path = osp.join(args.pred_dir, run_name, 'test/masks', filename)
            if not osp.isfile(npy_path):
                continue
            npy_image = np.load(npy_path)
            masks.append(npy_image)



        # Perform ensembling
        stacked_masks = np.stack(masks, axis=0)
        ensembled, certainty_score = get_certainty_score_one_mask(stacked_masks, target_mask=target_mask)

        if certainty_score >= args.threshold:
            # Save numpy prediction
            dest_image_path = osp.join(args.out_dir, filename)
            np.save(dest_image_path, ensembled.astype(np.uint8))
            num_good += 1

    
    print(f"Total number of masks: {len(filenames)}")
    print(f"Number of uncertain masks: {len(filenames) - num_good }")