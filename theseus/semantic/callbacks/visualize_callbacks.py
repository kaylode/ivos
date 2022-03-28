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
from theseus.utilities.analysis.analyzer import SemanticAnalyzer
from theseus.utilities.cuda import move_to

LOGGER = LoggerObserver.getLogger("main")

class VisualizerCallbacks(Callbacks):
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
        classnames = valset.classnames

        self.visualize_model(model, train_batch)
        self.params['trainer'].evaluate_epoch()
        self.visualize_gt(train_batch, val_batch, iters, classnames)
        self.analyze_gt(trainset, valset, iters)

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
                'inputs': images
            }
        }])

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
            'tag': "Sanitycheck/batch/train",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': iters
            }
        }])

        # Validation
        images = val_batch["inputs"]
        masks = val_batch['targets'].squeeze()

        batch = []
        for idx, (inputs, mask) in enumerate(zip(images, masks)): # iter through batch
            # iter through timestamp
            for t_input, t_mask in zip(inputs, mask):
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
        plt.legend(handles=patches, bbox_to_anchor=(-0.03, 1), loc="upper right", borderaxespad=0., 
                fontsize='large', ncol=(len(classnames)//10)+1)
        plt.tight_layout(pad=0)

        LOGGER.log([{
            'tag': "Sanitycheck/batch/val",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': iters
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
        last_outputs = logs['last_outputs']['out']
        
        images = last_batch["inputs"].squeeze().numpy() # (B, T, C, H, W) 
        masks = last_batch['gt'].permute(3,0,1,2).numpy() # (B, T, H, W) 
        guidemark = last_batch['info']['guidemark'] # (B, T, H, W) 
        iters = logs['iters']

        first = images[:guidemark, :, :, :]
        second = images[guidemark:, :, :, :]
        second = np.flip(second, axis=0)
        image_show = np.concatenate([second, first[1:,:,:,:]], axis=0)

        # iter through timestamp
        decode_masks = []
        decode_preds = []
        for mask, pred in zip(masks, last_outputs):
            decode_pred = self.visualizer.decode_segmap(pred)
            decode_mask = self.visualizer.decode_segmap(mask)
            decode_masks.append(decode_mask)
            decode_preds.append(decode_pred)
        decode_masks = np.stack(decode_masks, axis=0).transpose(0,3,1,2)
        decode_preds = np.stack(decode_preds, axis=0).transpose(0,3,1,2)

        LOGGER.log([{
            'tag': "Validation/visualization/inputs",
            'value': image_show,
            'type': LoggerObserver.VIDEO,
            'kwargs': {
                'step': iters,
                'fps': fps
            }
        }])

        LOGGER.log([{
            'tag': "Validation/visualization/ground truth",
            'value': decode_masks,
            'type': LoggerObserver.VIDEO,
            'kwargs': {
                'step': iters,
                'fps': fps
            }
        }])

        LOGGER.log([{
            'tag': "Validation/visualization/prediction",
            'value': decode_preds,
            'type': LoggerObserver.VIDEO,
            'kwargs': {
                'step': iters,
                'fps': fps
            }
        }])
