import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
import os.path as osp
import imageio
from tqdm import tqdm
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.semantic3D.utilities.preprocess.loading import (
    save_ct_from_npy,
    load_ct_info,
)
import torch

NUM_CLASSESS = 14

def encode_masks(masks):
    """
    Input masks from _load_mask(), but in shape [B, H, W]
    Output should be one-hot encoding of segmentation masks [B, NC, H, W]
    """

    one_hot = torch.nn.functional.one_hot(
        masks.long(), num_classes=NUM_CLASSESS
    )  # (B,H,W,NC)
    return one_hot.float()


def efficient_ensemble(list_of_masks, num_partritions=3):
    N, T, H, W = list_of_masks.shape
    list_of_masks = torch.from_numpy(list_of_masks)

    # Split along time dimension to avoid memory overload
    masks_splits = torch.split(list_of_masks, num_partritions, dim=1)

    result = []
    for m_split in masks_splits:
        one_hot_mask = encode_masks(m_split)
        one_hot_mask = torch.sum(one_hot_mask, dim=0)
        ensembled = torch.argmax(one_hot_mask, dim=-1)
        ensembled = ensembled.permute(1,2,0)
        result.append(ensembled)
    result = torch.cat(result, dim=-1)

    return result.numpy().astype(np.uint8)

def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols, f"{nindex} != {nrows} * {ncols}"
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

visualizer = Visualizer()
def visualize(list_of_masks, savedir, filename):    
    N = len(list_of_masks)
    T, H, W = list_of_masks[0].shape

    norm_images = []
    for i in range(T):
        vis_images = []
        for mask in list_of_masks:
            gt_mask = visualizer.decode_segmap(mask[i, :, :], NUM_CLASSESS)
            vis_images.append(gt_mask)
        image_show = gallery(np.stack(vis_images, axis=0), ncols=7)
        norm_images.append(image_show)
    norm_images = np.stack(norm_images, axis=0)
    imageio.mimsave(osp.join(savedir, f'{filename}.gif'), norm_images.astype(np.uint8))

if __name__ == "__main__":
    PRED_DIR = "runs/test_infer/"
    GT_DIR = "../data/nib_normalized/Validation"
    OUT_DIR = "runs/ensemble/new"
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_MASK_DIR = "runs/ensemble/new/masks"
    OUT_VIS_DIR = "runs/ensemble/new/vis"   
    os.makedirs(OUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUT_VIS_DIR, exist_ok=True)
    run_names = os.listdir(PRED_DIR)
    for run_name in run_names:
        print(run_name)
    filenames = os.listdir(GT_DIR)
    for filename in tqdm(filenames):
        filename = filename.replace("_0000.nii.gz", ".nii.gz")
        masks = []
        for i, run_name in enumerate(run_names):
            nib_path = osp.join(PRED_DIR, run_name, 'test/masks', filename)
            image_dict = load_ct_info(nib_path)
            masks.append(image_dict['npy_image'])
            if i == 0:
                origin = image_dict["origin"]
                spacing = image_dict["spacing"]
                direction = image_dict["direction"]
        stacked_masks = np.stack(masks, axis=0)
        dest_image_path = osp.join(OUT_MASK_DIR, filename)
        ensembled = efficient_ensemble(stacked_masks)

        masks.append(ensembled.transpose(2,0,1))

        visualize(masks, OUT_VIS_DIR, osp.splitext(filename)[0])
        
        # nib_label = nib.load(nib_path)
        # affine = nib_label.affine
        # ni_img = nib.Nifti1Image(ensembled, affine)
        # nib.save(ni_img, dest_image_path)

        save_ct_from_npy(
            npy_image=ensembled,
            save_path=dest_image_path,
            origin=image_dict["origin"],
            spacing=image_dict["spacing"],
            direction=image_dict["direction"],
            sitk_type=sitk.sitkUInt8,
        )
        break