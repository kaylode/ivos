from typing import List, Optional, Tuple

import matplotlib as mpl
mpl.use("Agg")
from theseus.opt import Opts

import os.path as osp
import time
import numpy as np
from tqdm import tqdm
import torch
import nibabel as nib
from theseus.opt import Config
from theseus.semantic2D.models import MODEL_REGISTRY
from theseus.semantic2D.augmentations import TRANSFORM_REGISTRY
from theseus.semantic2D.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from theseus.utilities.loggers import LoggerObserver
from theseus.semantic2D.models.stcn.inference.inference_core import InferenceCore
from theseus.semantic2D.models.stcn.networks.eval_network import STCNEval
from theseus.base.pipeline import BaseTestPipeline

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

    def init_loading(self):
        self.weights = self.opt['global']['weights']

        # Performs input mapping such that stage 0 model can be loaded
        prop_saved = torch.load(self.weights)['model']
        for k in list(prop_saved.keys()):
            if k == 'value_encoder.conv1.weight':
                if prop_saved[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
                    prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
        self.prop_model.load_state_dict(prop_saved)
        
    @torch.no_grad()
    def inference(self):
        self.init_pipeline()
        self.logger.text("Inferencing...", level=LoggerObserver.INFO)
        torch.autograd.set_grad_enabled(False)
        
        total_process_time = 0
        total_frames = 0

        for data in tqdm(self.dataloader):

            with torch.cuda.amp.autocast(enabled=False):
                rgb = data['inputs'].float().cuda()
                msk = data['targets'][0].cuda()
                info = data['info']
                name = info['name']
                guidemark = info['guidemark']
                affine = info['affine']
                k = len(info['labels'][0])

                torch.cuda.synchronize()
                process_begin = time.time()

                processor = InferenceCore(
                    self.prop_model, rgb, k, 
                    top_k=self.top_k, 
                    mem_every=self.mem_every)
                out_masks = processor.get_prediction({
                    'rgb': rgb,
                    'msk': msk,
                    'prop_range': [(0, guidemark), (guidemark, rgb.shape[1])] # reference guide frame index, 0 because we already process in the dataset
                })['masks']

                torch.cuda.synchronize()
                total_process_time += time.time() - process_begin
                total_frames += out_masks.shape[0]

                first = out_masks[:guidemark, :, :]
                second = out_masks[guidemark:, :, :]
                second = np.flip(second, axis=0)

                out_masks = np.concatenate([second, first[1:,:,:]], axis=0)

                out_masks = out_masks.transpose(1,2,0) # H, W, T
                ni_img = nib.Nifti1Image(out_masks, affine.squeeze(0).numpy())
                patient_id = osp.basename(name[0]).split('.')[0].split('_')[0]
                this_out_path = osp.join(self.savedir, str(patient_id)+'.nii.gz')
                nib.save(ni_img, this_out_path)

            del rgb
            del msk
            del processor

        self.logger.text(f"Number of processed slices: {total_frames}", level=LoggerObserver.INFO)
        self.logger.text(f"Execution time: {total_process_time}s", level=LoggerObserver.INFO)
        

if __name__ == '__main__':
    opts = Opts().parse_args()
    val_pipeline = TestPipeline(opts)
    val_pipeline.inference()

        
