from typing import List, Optional, Tuple

import matplotlib as mpl

mpl.use("Agg")
from theseus.opt import Opts

import os
import os.path as osp
import cv2
import time
import numpy as np
from tqdm import tqdm
import torch
import imageio
from theseus.opt import Config
from theseus.utilities.loading import load_state_dict
from theseus.utilities.cuda import move_to
from theseus.utilities.getter import get_instance_recursively
from source.cps.models import MODEL_REGISTRY
from source.cps.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from source.semantic3D.augmentations import TRANSFORM_REGISTRY

from theseus.utilities.loggers import LoggerObserver
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.base.pipeline import BaseTestPipeline


class TestPipeline(BaseTestPipeline):
    def __init__(self, opt: Config):

        super(TestPipeline, self).__init__(opt)
        self.opt = opt

    def init_globals(self):
        super().init_globals()
        self.save_visualization = self.opt["global"]["save_visualization"]

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

            if hasattr(self.model.model1, 'model'):
                self.model.model1.model = load_state_dict(
                    self.model.model1.model, state_dict, "model1"
                )
            else:
                self.model.model1 = load_state_dict(
                    self.model.model1, state_dict, "model1"
                )
            
            if hasattr(self.model.model2, 'model'):
                self.model.model2.model = load_state_dict(
                    self.model.model2.model, state_dict, "model2"
                )
            else:
                self.model.model2 = load_state_dict(
                    self.model.model2, state_dict, "model2"
                )

    def save_gif(self, images, masks, save_dir, outname):
        # images: (T, C, H, W)
        # masks: (H, W, T)

        norm_images = []
        for i in range(images.shape[0]):
            norm_image = (images[i] - images[i].min()) / images[i].max()
            norm_image = (norm_image * 255).squeeze()
            norm_mask = self.visualizer.decode_segmap(masks[:, :, i], self.num_classes)
            # norm_image = np.stack([norm_image, norm_image, norm_image], axis=2)
            norm_image = norm_image.transpose(1,2,0)
            image_show = np.concatenate([norm_image, norm_mask], axis=-2)
            norm_images.append(image_show)

        norm_images = np.stack(norm_images, axis=0)

        imageio.mimsave(
            osp.join(save_dir, f"{outname}.gif"), norm_images.astype(np.uint8)
        )

    @torch.no_grad()
    def inference(self):
        self.init_pipeline()
        self.logger.text("Inferencing...", level=LoggerObserver.INFO)

        savedir = osp.join(self.savedir, "masks")
        os.makedirs(savedir, exist_ok=True)
        torch.autograd.set_grad_enabled(False)

        total_process_time = 0
        total_frames = 0

        self.classnames = self.dataset.classnames
        self.num_classes = len(self.classnames)

        df_dict = {"image": [], "label": []}

        for data in tqdm(self.dataloader):

            with torch.cuda.amp.autocast(enabled=False):
                # FIRST STAGE: Get reference frames
                process_begin = time.time()

                custom_batch = []
                out_masks = []
                inputs = data["ref_images"]
                sids = data['sids'][0]

                for i, (inp, sid) in enumerate(zip(inputs, sids)):
                    if len(custom_batch) == 31 or i == inputs.shape[0] - 1:
                        custom_batch.append((inp, sid))
                        with torch.no_grad():
                            batch_preds = self.model.get_prediction(
                                {
                                    "inputs": torch.stack([i[0] for i in custom_batch], dim=0),
                                    "sids": [i[1] for i in custom_batch]
                                },
                                self.device,
                            )["masks"]
                            custom_batch = []

                            if len(batch_preds.shape) == 2:
                                batch_preds = np.expand_dims(batch_preds, axis=0)
                        out_masks.append(batch_preds)
                    else:
                        custom_batch.append((inp, sid))
                out_masks = np.concatenate(out_masks, axis=0)

                name = data["infos"][0]["img_name"]
                ori_h, ori_w = data['infos'][0]['ori_size']

                torch.cuda.synchronize()
                total_process_time += time.time() - process_begin
                total_frames += out_masks.shape[0]

                resized_masks = []
                for mask in out_masks:
                    resized = cv2.resize(mask, tuple([ori_h, ori_w]), 0, 0, interpolation = cv2.INTER_NEAREST)
                    resized_masks.append(resized)
                resized_masks = np.stack(resized_masks, axis=0)

                resized_masks = resized_masks.transpose(1, 2, 0)  # H, W, T
                resized_masks = resized_masks.astype(np.uint8)

                this_out_path = osp.join(savedir, str(name).replace("_0000.nii.gz", ".npy"))
                np.save(this_out_path, resized_masks)

                df_dict["image"].append(name)
                df_dict["label"].append(osp.join("masks", name))

            if self.save_visualization:
                gif_name = str(name).split(".")[0]
                visdir = osp.join(self.savedir, "visualization")
                os.makedirs(visdir, exist_ok=True)
                self.save_gif(data["ref_images"].numpy(), out_masks.transpose(1, 2, 0).astype(np.uint8), visdir, gif_name)
                self.logger.text(f"Saved to {gif_name}", level=LoggerObserver.INFO)

        self.logger.text(
            f"Number of processed slices: {total_frames}", level=LoggerObserver.INFO
        )
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
