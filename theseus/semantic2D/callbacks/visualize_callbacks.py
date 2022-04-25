from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchvision.transforms import functional as TFF
import numpy as np
from theseus.utilities.visualization.colors import color_list
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.utilities.loggers.observer import LoggerObserver
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.semantic2D.callbacks.base_visualize_callbacks import BaseVisualizerCallbacks
from theseus.utilities.cuda import move_to

LOGGER = LoggerObserver.getLogger("main")

class VisualizerCallbacks(BaseVisualizerCallbacks):
    """
    Callbacks for visualizing stuff during training
    Features:
        - Visualize datasets; plot model architecture, analyze datasets in sanity check
        - Visualize prediction at every end of validation

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @torch.no_grad()
    def on_val_epoch_end(self, logs: Dict=None):
        """
        After finish validation
        """
        LOGGER.text("Visualizing predictions...", level=LoggerObserver.DEBUG)

        fps = 10
        last_batch = logs['last_batch']
        last_outputs = logs['last_outputs']['outputs']
        
        images = last_batch["inputs"].squeeze().numpy() # (B, T, C, H, W) 
        masks = last_batch['targets'].permute(3,0,1,2).long().numpy() # (B, T, H, W) 
        guidemark = last_batch['info']['guidemark']
        guide_id = last_batch['info']['guide_id']
        iters = logs['iters']

        first = images[:guidemark, :, :, :]
        second = images[guidemark:, :, :, :]
        second = np.flip(second, axis=0)
        image_show = np.concatenate([second, first[1:,:,:,:]], axis=0)
        
        # iter through timestamp
        vis_inputs = []
        for image in image_show:
            image = image.transpose(1,2,0)
            image = self.visualizer.denormalize(image, mean=[0,0,0], std=[1,1,1])
            image = (image*255).astype(int)
            vis_inputs.append(image)
        vis_inputs = np.stack(vis_inputs, axis=0).transpose(0,3,1,2)

        # iter through timestamp
        decode_masks = []
        decode_preds = []
        for mask, pred in zip(masks, last_outputs):
            decode_pred = self.visualizer.decode_segmap(pred)
            decode_mask = self.visualizer.decode_segmap(mask.squeeze())
            decode_masks.append(decode_mask)
            decode_preds.append(decode_pred)
        decode_masks = np.stack(decode_masks, axis=0).transpose(0,3,1,2)
        decode_preds = np.stack(decode_preds, axis=0).transpose(0,3,1,2)

        concated_vis = np.concatenate([vis_inputs, decode_masks, decode_preds], axis=-1) # (T, C, 3H, W)
        reference_img = concated_vis[guide_id].transpose(1,2,0) # (C, 3H, W)

        LOGGER.log([{
            'tag': "Validation/val_prediction",
            'value': concated_vis,
            'type': LoggerObserver.VIDEO,
            'kwargs': {
                'step': iters,
                'fps': fps
            }
        }])

        fig = plt.figure(figsize=(15,5))
        plt.axis('off')
        plt.imshow(reference_img)

        # segmentation color legends 
        patches = [mpatches.Patch(color=np.array(color_list[i][::-1]), 
                                label=self.classnames[i]) for i in range(len(self.classnames))]
        plt.legend(handles=patches, bbox_to_anchor=(-0.03, 1), loc="upper right", borderaxespad=0., 
                fontsize='large', ncol=(len(self.classnames)//10)+1)
        plt.tight_layout(pad=0)

        LOGGER.log([{
            'tag': "Validation/reference",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': iters
            }
        }])

class NormalVisualizerCallbacks(BaseVisualizerCallbacks):
    """
    Callbacks for visualizing stuff during training
    Features:
        - Visualize datasets; plot model architecture, analyze datasets in sanity check
        - Visualize prediction at every end of validation

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @torch.no_grad()
    def on_val_epoch_end(self, logs: Dict=None):
        """
        After finish validation
        """

        iters = logs['iters']
        last_batch = logs['last_batch']
        last_outputs = logs['last_outputs']['outputs']
        model = self.params['trainer'].model
        valloader = self.params['trainer'].valloader

        # Vizualize model predictions
        LOGGER.text("Visualizing model predictions...", level=LoggerObserver.DEBUG)

        model.eval()

        images = last_batch["inputs"]
        masks = last_batch['targets'].squeeze()

        preds = torch.argmax(last_outputs, dim=1)
        masks = torch.argmax(masks, dim=1)
        preds = move_to(preds, torch.device('cpu'))

        batch = []
        for idx, (inputs, mask, pred) in enumerate(zip(images, masks, preds)):
            img_show = self.visualizer.denormalize(inputs)
            decode_mask = self.visualizer.decode_segmap(mask.numpy())
            decode_pred = self.visualizer.decode_segmap(pred)
            img_cam = TFF.to_tensor(img_show)
            decode_mask = TFF.to_tensor(decode_mask/255.0)
            decode_pred = TFF.to_tensor(decode_pred/255.0)
            img_show = torch.cat([img_cam, decode_pred, decode_mask], dim=-1)
            batch.append(img_show)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.title('Raw image - Prediction - Ground Truth')
        plt.imshow(grid_img)

        # segmentation color legends 
        classnames = valloader.dataset.classnames
        patches = [mpatches.Patch(color=np.array(color_list[i][::-1]), 
                                label=classnames[i]) for i in range(len(classnames))]
        plt.legend(handles=patches, bbox_to_anchor=(-0.03, 1), loc="upper right", borderaxespad=0., 
                fontsize='large', ncol=(len(classnames)//10)+1)
        plt.tight_layout(pad=0)

        LOGGER.log([{
            'tag': "Validation/val_prediction",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': iters
            }
        }])

        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close()