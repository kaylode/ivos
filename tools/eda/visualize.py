import os
import cv2
import os.path as osp
from tqdm import tqdm
import nibabel as nib
import numpy as np
from theseus.utilities.visualization.visualizer import Visualizer

import argparse
parser = argparse.ArgumentParser("Perform data analysis on volume CT")
parser.add_argument("-i", "--image_dir", type=str, help="Volume directory")
parser.add_argument("-l", "--mask_dir", default=None, type=str, help="Volume directory")
parser.add_argument("-o", "--out_dir", type=str, help="Volume directory")

VISUALIZER = Visualizer()
NUM_CLASSES = 15    
class VideoWriter:
    def __init__(self, video_info, saved_path):
        self.video_info = video_info
        self.saved_path = saved_path

        os.makedirs(self.saved_path, exist_ok=True)
            
        video_name = self.video_info['name']
        outpath =os.path.join(self.saved_path, video_name)

        self.FPS = self.video_info['fps']
        self.WIDTH = self.video_info['width']
        self.HEIGHT = self.video_info['height']
        self.NUM_FRAMES = self.video_info['num_frames']
        self.outvid = cv2.VideoWriter(
            outpath,   
            cv2.VideoWriter_fourcc(*'mp4v'), 
            self.FPS, 
            (self.WIDTH, self.HEIGHT))

    def write_frame(self, frame):
        self.outvid.write(frame)


def normalize_min_max(array):
    norm_array = (array - np.min(array)) / np.max(array)
    return norm_array

def make_frames(image_path, mask_path=None):
    image = nib.load(image_path).get_fdata()# (H, W, NS)
    height, width, depth = image.shape
    if mask_path is not None:
        mask = nib.load(mask_path).get_fdata()# (H, W, NS)
        
    images= []
    for i in range(image.shape[-1]):
        norm_image = normalize_min_max(image[:, :, i])
        norm_image = np.stack([norm_image, norm_image, norm_image], axis=2)
        if mask_path is not None:
            norm_mask = VISUALIZER.decode_segmap(mask[:, :, i], NUM_CLASSES)
            
            norm_image = np.concatenate([norm_image, norm_mask[:,:,::-1]], axis=1)
        
        images.append(norm_image)

    return images, height, width, depth

def run(args):
    filenames = os.listdir(args.image_dir)

    for filename in tqdm(filenames):
        image_path = osp.join(args.image_dir, filename)
        maskname = filename.replace('_0000.nii.gz', '.nii.gz') 
        mask_path = osp.join(args.mask_dir, maskname)
        frames, height, width, depth = make_frames(image_path, mask_path)
        
        writer = VideoWriter({
            'name': osp.splitext(filename)[0]+'.mp4',
            'fps': 10,
            'width': width*2,
            'height': height,
            'num_frames': depth
        }, args.out_dir)

        for frame in frames:
            writer.write_frame((frame*255).astype(np.uint8))


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)