from typing import List, Optional, Tuple

import matplotlib as mpl
mpl.use("Agg")
from theseus.opt import Opts

import os
import os.path as osp
import time
import numpy as np
from tqdm import tqdm
import torch
import imageio
import nibabel as nib
from theseus.opt import Config
from theseus.semantic2D.models import MODEL_REGISTRY
from theseus.semantic3D.augmentations import TRANSFORM_REGISTRY
from theseus.semantic2D.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from theseus.utilities.loggers import LoggerObserver
from theseus.semantic2D.models.stcn.inference.inference_core import InferenceCore
from theseus.semantic2D.models.stcn.networks.eval_network import STCNEval
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.base.pipeline import BaseTestPipeline
from theseus.utilities.loading import load_state_dict
from theseus.utilities.getter import get_instance

class TestPipeline(BaseTestPipeline):
    def __init__(
            self,
            opt: Config
        ):

        super(TestPipeline, self).__init__(opt)
        self.opt = opt

    def init_globals(self):
        super().init_globals()
        self.top_k = self.opt['global']['top_k']
        self.mem_every = self.opt['global']['mem_every']
        self.save_visualization = self.opt['global']['save_visualization']

        if self.save_visualization:
            self.visualizer = Visualizer()


    def init_registry(self):
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.logger.text(
            "Overidding registry in pipeline...", LoggerObserver.INFO
        )

    def init_model(self):
        # Load our checkpoint
        self.prop_model = STCNEval().to(self.device).eval()

        self.classnames = self.dataset.classnames
        self.ref_model = get_instance(
            self.opt["ref_model"], 
            registry=self.model_registry, 
            num_classes=len(self.classnames),
            classnames=self.classnames)

        self.num_classes = len(self.classnames)

    def init_loading(self):
        self.prop_weights = self.opt['global']['prop_weights']
        self.ref_weights = self.opt['global']['ref_weights']
 

        # Performs input mapping such that stage 0 model can be loaded
        prop_saved = torch.load(self.prop_weights)['model']
        for k in list(prop_saved.keys()):
            if k == 'value_encoder.conv1.weight':
                if prop_saved[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
                    prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
        self.prop_model.load_state_dict(prop_saved)

        # Load reference model
        ref_state_dict = torch.load(self.ref_weights)
        self.ref_model.model = load_state_dict(self.ref_model.model, ref_state_dict, "model")
        self.ref_model = self.ref_model.to(self.device)
        self.ref_model.eval()

    def search_reference(self, vol_mask, global_indices, pad_length, strategy="all"):
        """
        vol_mask: argmaxed segmentation mask. (T//sr, H, W)
        pad_length: int, return reference masks in targeted length with padding
        """

        assert strategy in ['all', 'best', 'random'], "Wrong strategy chosen"
        num_slices = vol_mask.shape[0]

        # Search for guide frames, in which most classes are presented
        candidates_local_indices = []
        max_possible_number_of_classes = 0
        for frame_idx in range(num_slices):
            num_classes = len(np.unique(vol_mask[frame_idx, :, :]))
            if num_classes == max_possible_number_of_classes:
                candidates_local_indices.append(frame_idx)
            elif num_classes > max_possible_number_of_classes:
                max_possible_number_of_classes = num_classes
                candidates_local_indices = [frame_idx]
        
        
        if strategy == 'random':
            random_idx = np.random.choice(candidates_local_indices)
            candidates_local_indices = [random_idx]

        # Generate reference frame, contains most suitable annotation masks 
        candidates_global_indices = [global_indices[i] for i in candidates_local_indices]
        prop_range = [
            (candidates_global_indices[i], candidates_global_indices[i+1])
            for i in range(len(candidates_global_indices)-1)
        ]
        prop_range = \
            [(candidates_global_indices[0], 0)] \
            + prop_range \
            + [(candidates_global_indices[-1], pad_length)] 

        global_to_local = {
          k:v for k, v in zip(candidates_global_indices, candidates_local_indices)
        }

        masks = []
        for local_idx, global_idx in enumerate(range(pad_length)):
            if global_idx in candidates_global_indices:
                masks.append(vol_mask[global_to_local[global_idx]])
            else:
                masks.append(np.zeros_like(vol_mask[0]))
        
        masks = np.stack(masks, 0)
        masks = torch.from_numpy(masks)

        # for evaluation (H, W, num_slices)
        return masks, prop_range

    def _encode_masks(self, masks):
        """
        Input masks from _load_mask(), but in shape [B, H, W]
        Output should be one-hot encoding of segmentation masks [B, NC, H, W]
        """
        
        one_hot = torch.nn.functional.one_hot(masks.long(), num_classes=self.num_classes) # (B,H,W,NC)
        one_hot = one_hot.permute(0, 3, 1, 2) # (B,NC,H,W)
        return one_hot.float()

    def save_gif(self, images, masks, save_dir, outname):
        # images: (T, C, H, W)
        # masks: (H, W, T)
        if len(images.shape) == 4:
            images = images[:, 0, :, :]
        norm_images = []
        norm_masks = []
        for i in range(images.shape[0]):
            norm_image = (images[i, :, :] - images[i, :, :].min()) / images[i, :, :].max()
            norm_image = (norm_image*255)
            norm_mask = self.visualizer.decode_segmap(masks[:, :, i], self.num_classes)
            norm_images.append(norm_image)
            norm_masks.append(norm_mask)

        norm_images = np.stack(norm_images, axis = 0)
        norm_masks = np.stack(norm_masks, axis = 0)
        
        imageio.mimsave(osp.join(save_dir, f'{outname}_input.gif'), norm_images.astype(np.uint8))
        imageio.mimsave(osp.join(save_dir, f'{outname}_mask.gif'), norm_masks.astype(np.uint8))

    @torch.no_grad()
    def inference(self):
        self.init_pipeline()
        self.logger.text("Inferencing...", level=LoggerObserver.INFO)

        savedir = osp.join(self.savedir, 'nib')
        os.makedirs(savedir, exist_ok=True)
        torch.autograd.set_grad_enabled(False)
        
        total_process_time = 0
        total_frames = 0

        for data in tqdm(self.dataloader):
            
            with torch.cuda.amp.autocast(enabled=False):                
                # FIRST STAGE: Get reference frames
                with torch.no_grad():
                    candidates = self.ref_model.get_prediction({
                        'inputs': data['ref_images']
                    }, self.device)['masks']

                full_images = data['full_images']
                ref_frames, prop_range = self.search_reference(
                    candidates, 
                    global_indices=data['ref_indices'][0], 
                    pad_length=full_images.shape[1])

                
                # SECOND STAGE: Full images
                rgb = full_images.float()
                msk = self._encode_masks(ref_frames)
                k = self.num_classes
                msk = msk.permute(1,0,2,3).unsqueeze(2)
                info = data['infos'][0]
                name = info['img_name']
                affine = info['affine']
       
                torch.cuda.synchronize()
                process_begin = time.time()

                processor = InferenceCore(
                    self.prop_model, rgb, k, 
                    top_k=self.top_k, 
                    mem_every=self.mem_every)
                
                with torch.no_grad():
                    out_masks = processor.get_prediction({
                        'rgb': rgb,
                        'msk': msk[1:,...],
                        'prop_range': prop_range
                    })['masks']

                torch.cuda.synchronize()
                total_process_time += time.time() - process_begin
                total_frames += out_masks.shape[0]

                out_masks = out_masks.transpose(1,2,0) # H, W, T
                ni_img = nib.Nifti1Image(out_masks, affine)
                this_out_path = osp.join(savedir, str(name))
                nib.save(ni_img, this_out_path)

            del rgb
            del msk
            del processor

            if self.save_visualization:
                gif_name = str(name).split('.')[0]
                visdir = osp.join(self.savedir, 'visualization')
                os.makedirs(visdir, exist_ok=True)
                self.save_gif(full_images.squeeze(), out_masks, visdir, gif_name)
                self.logger.text(f"Saved to {gif_name}", level=LoggerObserver.INFO)

        self.logger.text(f"Number of processed slices: {total_frames}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of processed volumes: {len(self.dataloader)}", level=LoggerObserver.INFO)
        self.logger.text(f"Execution time: {total_process_time}s", level=LoggerObserver.INFO)
        

if __name__ == '__main__':
    opts = Opts().parse_args()
    val_pipeline = TestPipeline(opts)
    val_pipeline.inference()