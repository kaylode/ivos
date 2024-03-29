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
from theseus.utilities.cuda import move_to

LOGGER = LoggerObserver.getLogger("main")

class VolumeVisualizerCallbacks(Callbacks):
    """
    Callbacks for visualizing stuff during training
    Features:
        - Visualize datasets; plot model architecture, analyze datasets in sanity check
        - Visualize prediction at every end of validation
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.visualizer = Visualizer()

    def sanitycheck(self, logs: Dict=None):
        """
        Sanitycheck before starting. Run only when debug=True
        """

        iters = logs['iters']
        model = self.params['trainer'].model
        valloader = self.params['trainer'].valloader
        trainloader = self.params['trainer'].trainloader
        train_batch = next(iter(trainloader))
        val_batch = next(iter(valloader))
        trainset = trainloader.dataset
        valset = valloader.dataset
        self.classnames = valset.classnames

        self.visualize_model(model, train_batch)
        self.visualize_gt(train_batch, val_batch, iters, self.classnames)
        self.analyze_gt(trainset, valset, iters)
        self.params['trainer'].evaluate_epoch()

    @torch.no_grad()
    def visualize_model(self, model, batch):
        # Vizualize Model Graph
        LOGGER.text("Visualizing architecture...", level=LoggerObserver.DEBUG)
        images = move_to(batch["inputs"], model.device)
        LOGGER.log([{
            'tag': "Sanitycheck/analysis/architecture",
            'value': model.model.get_model(),
            'type': LoggerObserver.TORCH_MODULE,
            'kwargs': {
                'inputs': images,
                'log_freq': 100
            }
        }])

    def normalize_min_max(self, array):
        norm_array = (array - array.min()) / array.max()
        return norm_array

    def visualize_gt(self, train_batch, val_batch, iters, classnames):
        """
        Visualize dataloader for sanity check 
        """

        LOGGER.text("Visualizing dataset...", level=LoggerObserver.DEBUG)
        images = train_batch["inputs"] # (B, T, C, H, W) 
        masks = train_batch['cls_gt'].squeeze() # (B, T, H, W) 

        batch = []
        for idx, (inputs, mask) in enumerate(zip(images, masks)): # iter through batch
            # iter through timestamp
            for t_input, t_mask in zip(inputs, mask):
                t_input = self.normalize_min_max(t_input)
                if len(t_input.shape) == 2:
                    t_input = torch.stack([t_input, t_input, t_input], dim=1)
                img_show = self.visualizer.denormalize(t_input, mean=[0,0,0], std=[1,1,1])
                decode_mask = self.visualizer.decode_segmap(t_mask.numpy())
                img_show = TFF.to_tensor(img_show)
                decode_mask = TFF.to_tensor(decode_mask/255.0)
                img_show = torch.cat([img_show, decode_mask], dim=-1)
                batch.append(img_show)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.imshow(grid_img)

        # segmentation color legends 
        patches = [mpatches.Patch(color=np.array(color_list[i][::-1]), 
                                label=classnames[i]) for i in range(len(classnames))]
        plt.legend(handles=patches, bbox_to_anchor=(-0.03, 1), loc="upper right", borderaxespad=0., 
                fontsize='large', ncol=(len(classnames)//10)+1)
        plt.tight_layout(pad=0)

        LOGGER.log([{
            'tag': "Sanitycheck/train_batch",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': iters
            }
        }])


        image_show = val_batch["inputs"].squeeze().numpy() # (B, T, C, H, W) 
        masks = val_batch['targets'].permute(3,0,1,2).long().numpy() # (B, T, H, W) 
      
        # iter through timestamp
        vis_inputs = []
        for image in image_show:
            if len(image.shape) == 2:
                image = self.normalize_min_max(image)
                image = self.visualizer.denormalize(image, mean=[0], std=[1])
                image = (image*255).astype(int)
                image = np.stack([image, image, image], axis=-1)
            else:
                image = image.transpose(1,2,0)
            vis_inputs.append(image)
        vis_inputs = np.stack(vis_inputs, axis=0).transpose(0,3,1,2)

        # iter through timestamp
        decode_masks = []
        for mask in masks:
            decode_mask = self.visualizer.decode_segmap(mask.squeeze())
            decode_masks.append(decode_mask)
        decode_masks = np.stack(decode_masks, axis=0).transpose(0,3,1,2)

        concated_vis = np.concatenate([vis_inputs, decode_masks], axis=-1)
        
        ###
        LOGGER.log([{
            'tag': "Sanitycheck/val_batch",
            'value': concated_vis,
            'type': LoggerObserver.VIDEO,
            'kwargs': {
                'step': iters,
                'fps': 10
            }
        }])

        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close()

    def analyze_gt(self, trainset, valset, iters):
        """
        Perform simple data analysis
        """

        LOGGER.text("Analyzing datasets...", level=LoggerObserver.DEBUG)
        return

    @torch.no_grad()
    def on_val_epoch_end(self, logs: Dict=None):
        """
        After finish validation
        """
        LOGGER.text("Visualizing predictions...", level=LoggerObserver.DEBUG)

        fps = 10
        last_batch = logs['last_batch']
        last_outputs = logs['last_outputs']['outputs']
        
        image_show = last_batch["inputs"].squeeze().numpy() # (B, T, C, H, W) 
        masks = last_batch['targets'].permute(3,0,1,2).long().numpy() # (B, T, H, W) 
        iters = logs['iters']

        # iter through timestamp
        vis_inputs = []
        for image in image_show:
            if len(image.shape) == 2:
                image = self.normalize_min_max(image)
                image = self.visualizer.denormalize(image, mean=[0], std=[1])
                image = (image*255).astype(int)
                image = np.stack([image, image, image], axis=-1)
            else:
                image = image.transpose(1,2,0)
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

        LOGGER.log([{
            'tag': "Validation/val_prediction",
            'value': concated_vis,
            'type': LoggerObserver.VIDEO,
            'kwargs': {
                'step': iters,
                'fps': fps
            }
        }])