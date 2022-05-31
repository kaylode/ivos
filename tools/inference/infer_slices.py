from typing import List, Optional, Tuple

import matplotlib as mpl

mpl.use("Agg")
from theseus.opt import Opts

import os
import os.path as osp
import time
import numpy as np
from tqdm import tqdm
import cv2
import torch
import imageio
import numpy as np
import pandas as pd
from theseus.opt import Config
from theseus.utilities.loading import load_state_dict
from theseus.utilities.cuda import move_to
from theseus.utilities.getter import get_instance_recursively
from theseus.cps.models import MODEL_REGISTRY
from theseus.semantic3D.augmentations import TRANSFORM_REGISTRY
from theseus.semantic2D.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from theseus.utilities.loggers import LoggerObserver
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.base.pipeline import BaseTestPipeline


class TestPipeline(BaseTestPipeline):
    def __init__(self, opt: Config):

        super(TestPipeline, self).__init__(opt)
        self.opt = opt

    def init_globals(self):
        super().init_globals()
        self.save_csv = self.opt["global"]["save_csv"]
        self.use_uncertainty = self.opt["global"]["use_uncertainty"]
        self.save_visualization = self.opt["global"]["save_visualization"]
        self.class_weights = self.opt["global"]["class_weights"]

        if self.save_visualization:
            self.visualizer = Visualizer()

    def init_registry(self):
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.logger.text("Overidding registry in pipeline...", LoggerObserver.INFO)

    def init_model(self):
        CLASSNAMES = self.dataset.classnames
        self.model = get_instance_recursively(
            self.opt["model"],
            registry=self.model_registry,
            num_classes=len(CLASSNAMES),
            classnames=CLASSNAMES,
        )
        self.model = move_to(self.model, self.device)
        self.model.eval()

    def init_loading(self):
        self.weights = self.opt["global"]["weights"]
        if self.weights:
            state_dict = torch.load(self.weights)
            self.model.model1.model = load_state_dict(
                self.model.model1.model, state_dict, "model1"
            )
            self.model.model2.model = load_state_dict(
                self.model.model2.model, state_dict, "model2"
            )

    def load_class_weights(self):
        with open(self.class_weights, 'r') as f:
            class_weights = f.read().splitlines()
            class_weights = [float(i) for i in class_weights]
        return class_weights

    @torch.no_grad()
    def inference(self):
        self.init_pipeline()
        self.logger.text("Inferencing...", level=LoggerObserver.INFO)

        visualizer = Visualizer()

        saved_mask_dir = os.path.join(self.savedir, 'masks')
        saved_overlay_dir = os.path.join(self.savedir, 'vis')

        os.makedirs(saved_mask_dir, exist_ok=True)
        os.makedirs(saved_overlay_dir, exist_ok=True)
        df_dict = {"image": [], "label": []}
        total_process_time = 0

        if self.class_weights is not None:
            class_weights = self.load_class_weights()

        for idx, batch in enumerate(self.dataloader):
            inputs = batch['inputs']
            img_names = batch['img_names']
            ori_sizes = batch['ori_sizes']

            if self.class_weights is not None:
                batch.update({'weights': class_weights})
            
            start_time = time.time()
            outputs = self.model.get_prediction(batch, self.device)
            end_time = time.time()
            total_process_time += end_time - start_time
            preds = outputs['masks']

            for (inpt, pred, filename, ori_size) in zip(inputs, preds, img_names, ori_sizes):
                resized_pred = cv2.resize(pred, tuple(ori_size))

                # Save mask
                id = osp.splitext(filename)[0]  # FLARE22_Tr_0001_0000_0009
                pid = "_".join(id.split("_")[:-1])
                os.makedirs(os.path.join(saved_mask_dir, pid), exist_ok=True)
                savepath = os.path.join(saved_mask_dir, pid, filename)
                np.save(savepath, resized_pred.astype(np.uint8))

                # Save overlay
                ori_image = cv2.resize(inpt.squeeze().numpy(), tuple(ori_size))
                norm_image = (ori_image - ori_image.min()) / ori_image.max()
                norm_image = (norm_image*255).astype(np.uint8)
                decode_pred = visualizer.decode_segmap(pred)[:,:,::-1]

                norm_image = np.stack([norm_image, norm_image, norm_image], axis=-1)
                viz = np.concatenate([norm_image, decode_pred], axis=-2) 
                savepath = os.path.join(saved_overlay_dir, filename[:-4]+'.jpg')
                cv2.imwrite(savepath, viz.astype(np.uint8))

                self.logger.text(f"Save image at {savepath}", level=LoggerObserver.INFO)

                df_dict["image"].append(filename)
                df_dict["label"].append(osp.join("masks", filename))

        if self.save_csv:
            pd.DataFrame(df_dict).to_csv(
                osp.join(self.savedir, "pseudo.csv"), index=False
            )
            self.logger.text(f"Save csv at pseudo.csv", level=LoggerObserver.INFO)


        self.logger.text(
            f"Number of processed volumes: {len(self.dataloader)}",
            level=LoggerObserver.INFO,
        )
        self.logger.text(
            f"Execution time: {total_process_time}s", level=LoggerObserver.INFO
        )


if __name__ == "__main__":
    opts = Opts().parse_args()
    val_pipeline = TestPipeline(opts)
    val_pipeline.inference()
