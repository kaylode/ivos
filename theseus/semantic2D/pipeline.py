from typing import Callable, Dict, Optional
import torch
from theseus.opt import Config
from theseus.utilities.getter import (get_instance, get_instance_recursively)
from theseus.utilities.cuda import get_devices_info

from theseus.opt import Config
from theseus.base.pipeline import BasePipeline
from theseus.semantic2D.models.wrapper import ModelWithLoss
from theseus.base.optimizers import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from theseus.semantic2D.augmentations import TRANSFORM_REGISTRY
from theseus.semantic2D.losses import LOSS_REGISTRY
from theseus.semantic2D.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.semantic2D.trainer import TRAINER_REGISTRY
from theseus.semantic2D.metrics import METRIC_REGISTRY
from theseus.semantic2D.models import MODEL_REGISTRY
from theseus.semantic2D.callbacks import CALLBACKS_REGISTRY
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
        super().init_loading()
        if self.pretrained:
            state_dict = torch.load(self.pretrained, map_location='cpu')
            self.model.model.load_network(state_dict['model'])

        if self.resume:
            state_dict = torch.load(self.resume, map_location='cpu')
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

    def init_metrics(self):
        CLASSNAMES = self.val_dataset.classnames
        self.metrics = get_instance_recursively(
            self.opt['metrics'], 
            registry=self.metric_registry, 
            num_classes=len(CLASSNAMES),
            classnames=CLASSNAMES)

    def init_model_with_loss(self):
        model = self.init_model()
        criterion = self.init_criterion()
        self.model = ModelWithLoss(model, criterion, self.device)
        self.logger.text(f"Number of trainable parameters: {self.model.trainable_parameters():,}", level=LoggerObserver.INFO)
        device_info = get_devices_info(self.device_name)
        self.logger.text("Using " + device_info, level=LoggerObserver.INFO)