import numpy as np
import os
import os.path as osp
import imageio
from tqdm import tqdm
from theseus.utilities.visualization.visualizer import Visualizer
import torch
import argparse

parser = argparse.ArgumentParser(
    "Ensemble all prediction into one"
)
parser.add_argument(
    "-p", "--pred_dir", type=str, help="Volume directory contains prediction numpy images"
)
parser.add_argument("-o", "--out_dir", type=str, help="Output directory")

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
    N, H, W, T = list_of_masks.shape
    list_of_masks = torch.from_numpy(list_of_masks)

    # Split along time dimension to avoid memory overload
    masks_splits = torch.split(list_of_masks, T//num_partritions, dim=-1)
    result = []
    for m_split in masks_splits:
        one_hot_mask = encode_masks(m_split)
        one_hot_mask = torch.sum(one_hot_mask, dim=0)
        ensembled = torch.argmax(one_hot_mask, dim=-1)
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
    H, W, T = list_of_masks[0].shape

    norm_images = []    
    for i in range(T):
        vis_images = []
        for mask in list_of_masks:
            gt_mask = visualizer.decode_segmap(mask[:, :, i], NUM_CLASSESS)
            vis_images.append(gt_mask)  
        image_show = gallery(np.stack(vis_images, axis=0), ncols=N)
        norm_images.append(image_show)
    norm_images = np.stack(norm_images, axis=0)
    imageio.mimsave(osp.join(savedir, f'{filename}.gif'), norm_images.astype(np.uint8))

if __name__ == "__main__":
    args = parser.parse_args()
    OUT_MASK_DIR = osp.join(args.out_dir, "masks")
    OUT_VIS_DIR = osp.join(args.out_dir, "vis")   
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(OUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUT_VIS_DIR, exist_ok=True)

    print("Ensemble these runs:")
    run_names = os.listdir(args.pred_dir)
    for run_name in run_names:
        print(run_name)

    filenames = os.listdir(osp.join(args.pred_dir, run_name, 'test/masks'))
    for filename in tqdm(filenames):
        masks = []
        for i, run_name in enumerate(run_names):
            npy_path = osp.join(args.pred_dir, run_name, 'test/masks', filename)
            npy_image = np.load(npy_path)
            masks.append(npy_image)

        # Perform ensembling
        stacked_masks = np.stack(masks, axis=0)
        ensembled = efficient_ensemble(stacked_masks)
        
        # Visualization
        masks.append(ensembled)
        visualize(masks, OUT_VIS_DIR, osp.splitext(filename)[0])
        
        # Save numpy prediction
        dest_image_path = osp.join(OUT_MASK_DIR, filename)
        np.save(dest_image_path, ensembled.astype(np.uint8))
