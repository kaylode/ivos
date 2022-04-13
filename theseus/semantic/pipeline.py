from typing import Callable, Dict, Optional
import torch
from theseus.opt import Config
from theseus.utilities.getter import (get_instance)
from theseus.utilities.loading import load_state_dict
from theseus.utilities.cuda import get_devices_info

from theseus.opt import Config
from theseus.base.pipeline import BasePipeline
from theseus.semantic.models.wrapper import ModelWithLoss
from theseus.base.optimizers import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from theseus.semantic.augmentations import TRANSFORM_REGISTRY
from theseus.semantic.losses import LOSS_REGISTRY
from theseus.semantic.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.semantic.trainer import TRAINER_REGISTRY
from theseus.semantic.metrics import METRIC_REGISTRY
from theseus.semantic.models import MODEL_REGISTRY
from theseus.semantic.callbacks import CALLBACKS_REGISTRY
from theseus.utilities.loggers import LoggerObserver


class Pipeline(BasePipeline):
    """docstring for Pipeline."""

    def __init__(
        self,
        opt: Config
    ):
        super(Pipeline, self).__init__(opt)
        self.opt = opt

    def init_registry(self):
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.metric_registry = METRIC_REGISTRY
        self.loss_registry = LOSS_REGISTRY
        self.optimizer_registry = OPTIM_REGISTRY
        self.scheduler_registry = SCHEDULER_REGISTRY
        self.callbacks_registry = CALLBACKS_REGISTRY
        self.trainer_registry = TRAINER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.logger.text(
            "Overidding registry in pipeline...", LoggerObserver.INFO
        )

    def init_loading(self):
        self.resume = self.opt['global']['resume']
        self.pretrained = self.opt['global']['pretrained']
        self.last_epoch = -1
        if self.pretrained:
            state_dict = torch.load(self.pretrained, map_location='cpu')
            self.model.model.train_model = load_state_dict(self.model.model.train_model, state_dict)
            self.model.model.load_network(state_dict)

        if self.resume:
            state_dict = torch.load(self.resume, map_location='cpu')
            self.model.model.model = load_state_dict(self.model.model.model, state_dict, 'model')
            self.optimizer = load_state_dict(self.optimizer, state_dict, 'optimizer')
            iters = load_state_dict(None, state_dict, 'iters')
            self.last_epoch = iters//len(self.train_dataloader) - 1
            self.model.model.load_network(state_dict['model'])

    def init_model(self):
        CLASSNAMES = self.val_dataset.classnames
        model = get_instance(
            self.opt["model"], 
            registry=self.model_registry, 
            num_classes=len(CLASSNAMES),
            classnames=CLASSNAMES)
        # model = move_to(model, self.device)
        return model

    def init_model_with_loss(self):
        model = self.init_model()
        criterion = self.init_criterion()
        self.model = ModelWithLoss(model, criterion, self.device)
        self.logger.text(f"Number of trainable parameters: {self.model.trainable_parameters():,}", level=LoggerObserver.INFO)
        device_info = get_devices_info(self.device_name)
        self.logger.text("Using " + device_info, level=LoggerObserver.INFO)