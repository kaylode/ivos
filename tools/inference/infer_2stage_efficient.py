from theseus.utilities.getter import get_instance, get_instance_recursively
from theseus.utilities.loading import load_state_dict
from theseus.base.pipeline import BaseTestPipeline
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.semantic2D.models.stcn.networks.eval_network import STCNEval
from theseus.semantic2D.models.stcn.inference.inference_core_efficient import (
    InferenceCore,
)
from theseus.utilities.loggers import LoggerObserver
from theseus.semantic2D.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.semantic3D.augmentations import TRANSFORM_REGISTRY
from theseus.cps.models import MODEL_REGISTRY
from theseus.opt import Config
import imageio
import torch
from tqdm import tqdm
import numpy as np
import time
import os.path as osp
import os
import cv2
from theseus.opt import Opts
from typing import List, Optional, Tuple

import matplotlib as mpl

mpl.use("Agg")

from theseus.semantic2D.utilities.referencer import Referencer

REFERENCER = Referencer()


class TestPipeline(BaseTestPipeline):
    def __init__(self, opt: Config):

        super(TestPipeline, self).__init__(opt)
        self.opt = opt

    def init_globals(self):
        super().init_globals()
        self.reference_strategy = self.opt["global"]["ref_strategy"]
        self.propagation_strategy = self.opt["global"]["prop_strategy"]
        self.prop_config = self.opt["prop_model"]
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
        # Load our checkpoint
        self.prop_model = (
            STCNEval(
                key_backbone=self.prop_config["key_backbone"],
                value_backbone=self.prop_config["value_backbone"],
                pretrained=False,
            )
            .to(self.device)
            .eval()
        )

        self.classnames = self.dataset.classnames
        self.ref_model = get_instance_recursively(
            self.opt["ref_model"],
            registry=self.model_registry,
            num_classes=len(self.classnames),
            classnames=self.classnames,
        )

        self.num_classes = len(self.classnames)

    def init_loading(self):
        self.prop_weights = self.opt["global"]["prop_weights"]
        self.ref_weights = self.opt["global"]["ref_weights"]

        # Performs input mapping such that stage 0 model can be loaded
        prop_saved = torch.load(self.prop_weights)["model"]
        for k in list(prop_saved.keys()):
            if k == "value_encoder.conv1.weight":
                if prop_saved[k].shape[1] == 2:
                    pads = torch.zeros((64, 1, 7, 7), device=prop_saved[k].device)
                    prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
        self.prop_model.load_state_dict(prop_saved)

        # Load reference model
        ref_state_dict = torch.load(self.ref_weights)

        self.ref_model.model1 = load_state_dict(
            self.ref_model.model1, ref_state_dict, "model1"
        )
        self.ref_model.model2.model = load_state_dict(
            self.ref_model.model2.model, ref_state_dict, "model2"
        )
        self.ref_model = self.ref_model.to(self.device)
        self.ref_model.eval()

    def search_reference(self, vol_mask, global_indices, pad_length, strategy="all"):
        """
        vol_mask: argmaxed segmentation mask. (T//sr, H, W)
        pad_length: int, return reference masks in targeted length with padding
        """

        masks, indices = REFERENCER.search_reference_and_pack(
            vol_mask, global_indices, pad_length, strategy=strategy
        )

        # for evaluation (H, W, num_slices)
        return masks, indices

    def _encode_masks(self, masks):
        """
        Input masks from _load_mask(), but in shape [B, H, W]
        Output should be one-hot encoding of segmentation masks [B, NC, H, W]
        """

        one_hot = torch.nn.functional.one_hot(
            masks.long(), num_classes=self.num_classes
        )  # (B,H,W,NC)
        one_hot = one_hot.permute(0, 3, 1, 2)  # (B,NC,H,W)
        return one_hot.float()

    def save_gif(self, images, masks, save_dir, outname):
        # images: (T, C, H, W)
        # masks: (H, W, T)

        norm_images = []
        for i in range(images.shape[0]):
            norm_image = (images[i] - images[i].min()) / images[i].max()
            norm_image = (norm_image * 255).squeeze()
            norm_mask = self.visualizer.decode_segmap(masks[:, :, i], self.num_classes)
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

        for data in tqdm(self.dataloader):

            with torch.cuda.amp.autocast(enabled=False):
                # FIRST STAGE: Get reference frames

                inputs = data["ref_images"]
                sids = data['sids'][0]
                full_images = data['full_images'][0]

                custom_batch = []
                candidates = []
                with torch.no_grad():
                    for i, (inp, sid) in enumerate(zip(inputs, sids)):
                        if len(custom_batch) == 31 or i == inputs.shape[0] - 1:
                            custom_batch.append((inp, sid))
                            with torch.no_grad():
                                batch_preds = self.ref_model.get_prediction(
                                    {
                                        "inputs": torch.stack([i[0] for i in custom_batch], dim=0),
                                        "sids": [i[1] for i in custom_batch]
                                    },
                                    self.device,
                                )["masks"]
                                custom_batch = []

                                if len(batch_preds.shape) == 2:
                                    batch_preds = np.expand_dims(batch_preds, axis=0)
                            candidates.append(batch_preds)
                        else:
                            custom_batch.append((inp, sid))
                    candidates = np.concatenate(candidates, axis=0)

                ref_frames, ref_indices = self.search_reference(
                    candidates,
                    global_indices=data["ref_indices"][0],
                    pad_length=full_images.shape[0],
                    strategy=self.reference_strategy,
                )

                # SECOND STAGE: Full images
                rgb = full_images.unsqueeze(0).float()
                k = self.num_classes
                info = data["infos"][0]
                name = info["img_name"]

                torch.cuda.synchronize()
                process_begin = time.time()

                processor = InferenceCore(
                    self.prop_model,
                    rgb,
                    k,
                    top_k=self.prop_config["top_k"],
                    max_k=self.prop_config['max_k'],
                    mem_every=self.prop_config["mem_every"],
                    include_last=self.prop_config["include_last"],
                    device=self.device,
                )

                with torch.no_grad():
                    out_masks = processor.get_prediction(
                        {
                            "rgb": rgb,
                            "msk": ref_frames,
                            "guide_indices": ref_indices,
                            "bidirectional": self.prop_config["bidirectional"],
                        }
                    )["masks"]


                torch.cuda.synchronize()
                total_process_time += time.time() - process_begin
                total_frames += out_masks.shape[0]

                name = data["infos"][0]["img_name"]
                ori_h, ori_w = data['infos'][0]['ori_size']

                resized_masks = []
                for mask in out_masks:
                    resized = cv2.resize(mask, tuple([ori_h, ori_w]), 0, 0, interpolation = cv2.INTER_NEAREST)
                    resized_masks.append(resized)
                resized_masks = np.stack(resized_masks, axis=0)

                resized_masks = resized_masks.transpose(1, 2, 0)  # H, W, T
                resized_masks = resized_masks.astype(np.uint8)

                this_out_path = osp.join(savedir, str(name).replace("_0000.nii.gz", ".npy"))
                np.save(this_out_path, resized_masks)

            del rgb
            del processor

            if self.save_visualization:
                gif_name = str(name).split(".")[0]
                visdir = osp.join(self.savedir, "visualization")
                os.makedirs(visdir, exist_ok=True)
                self.save_gif(data["full_images"][0].numpy(), out_masks.transpose(1, 2, 0).astype(np.uint8), visdir, gif_name)
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
