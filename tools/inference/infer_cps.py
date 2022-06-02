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

    def save_gif(self, images, masks, save_dir, outname):
        # images: (T, C, H, W)
        # masks: (H, W, T)

        norm_images = []
        for i in range(images.shape[0]):
            norm_image = (images[i] - images[i].min()) / images[i].max()
            norm_image = (norm_image * 255).squeeze()
            norm_mask = self.visualizer.decode_segmap(masks[:, :, i], self.num_classes)
            norm_image = np.stack([norm_image, norm_image, norm_image], axis=2)
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
                inputs = data["inputs"]
                inputs = torch.rot90(inputs, 3, dims=(-2,-1))
                inputs = torch.flip(inputs, dims=(-1,))

                # import matplotlib.pyplot as plt
                # plt.imshow(inputs[40].squeeze())
                # plt.savefig('test')

                for i, inp in enumerate(inputs):
                    if len(custom_batch) == 0 or i == inputs.shape[0] - 1:
                        custom_batch.append(inp)
                        with torch.no_grad():
                            batch_preds = self.model.get_prediction(
                                {"inputs": torch.stack(custom_batch, dim=0)},
                                self.device,
                            )["masks"]
                            custom_batch = []

                            if len(batch_preds.shape) == 2:
                                batch_preds = np.expand_dims(batch_preds, axis=0)
                        out_masks.append(batch_preds)
                    else:
                        custom_batch.append(inp)
                out_masks = np.concatenate(out_masks, axis=0)

                out_masks = np.flip(out_masks, axis=-1)
                out_masks = np.rot90(out_masks, 1, axes=(-2,-1))

                # import matplotlib.pyplot as plt
                # plt.imshow(out_masks[40].squeeze())
                # plt.savefig('pred')
                # asd

                affine = data["affines"][0]
                name = data["img_names"][0]

                torch.cuda.synchronize()
                total_process_time += time.time() - process_begin
                total_frames += out_masks.shape[0]

                out_masks = out_masks.transpose(1, 2, 0)  # H, W, T
                out_masks = out_masks.astype(np.uint8)

                ni_img = nib.Nifti1Image(out_masks, affine)
                this_out_path = osp.join(savedir, str(name).replace("_0000.nii.gz", ".nii.gz"))
                nib.save(ni_img, this_out_path)

                df_dict["image"].append(name)
                df_dict["label"].append(osp.join("masks", name))

            if self.save_visualization:
                gif_name = str(name).split(".")[0]
                visdir = osp.join(self.savedir, "visualization")
                os.makedirs(visdir, exist_ok=True)
                self.save_gif(data["inputs"].numpy(), out_masks, visdir, gif_name)
                self.logger.text(f"Saved to {gif_name}", level=LoggerObserver.INFO)

        if self.save_csv:
            pd.DataFrame(df_dict).to_csv(
                osp.join(self.savedir, "pseudo.csv"), index=False
            )

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
