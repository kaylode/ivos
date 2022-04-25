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

class NormalVisualizerCallbacks(BaseVisualizerCallbacks):
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
        self.params['trainer'].evaluate_epoch()
        self.visualize_gt(train_batch, val_batch, iters, self.classnames)
        self.analyze_gt(trainset, valset, iters)

    @torch.no_grad()
    def visualize_model(self, model, batch):
        # Vizualize Model Graph
        LOGGER.text("Visualizing architecture...", level=LoggerObserver.DEBUG)
        LOGGER.log([{
            'tag': "Sanitycheck/analysis/architecture",
            'value': model.model.get_model(),
            'type': LoggerObserver.TORCH_MODULE,
            'kwargs': {
                'inputs': batch,
                'log_freq': 100
            }
        }])

    def visualize_gt(self, train_batch, val_batch, iters, classnames):
        """
        Visualize dataloader for sanity check 
        """

        LOGGER.text("Visualizing dataset...", level=LoggerObserver.DEBUG)
        images = train_batch["inputs"]
        masks = train_batch['targets'].squeeze()

        batch = []
        for idx, (inputs, mask) in enumerate(zip(images, masks)):
            img_show = inputs.squeeze().numpy()
            decode_mask = self.visualizer.decode_segmap(mask.numpy())
            img_show = TFF.to_tensor(img_show).repeat(3,1,1)
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

        # Validation
        images = val_batch["inputs"]
        masks = val_batch['targets'].squeeze()

        batch = []
        for idx, (inputs, mask) in enumerate(zip(images, masks)):
            img_show = inputs.squeeze().numpy()
            decode_mask = self.visualizer.decode_segmap(mask.numpy())
            img_show = TFF.to_tensor(img_show).repeat(3,1,1)
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
            'tag': "Sanitycheck/val_batch",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': iters
            }
        }])

        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close()

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
            img_show = inputs.squeeze().numpy()
            decode_mask = self.visualizer.decode_segmap(mask.numpy())
            decode_pred = self.visualizer.decode_segmap(pred.numpy())
            img_cam = TFF.to_tensor(img_show).repeat(3,1,1)
            decode_mask = TFF.to_tensor(decode_mask/255.0)
            decode_pred = TFF.to_tensor(decode_pred/255.0)
            img_show = torch.cat([img_cam, decode_pred, decode_mask], dim=-1)
            batch.append(img_show)
            if idx == 32:
                break
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