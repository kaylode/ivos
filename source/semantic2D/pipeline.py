from typing import Callable, Dict, Optional
import os
import torch
from theseus.opt import Config
from theseus.utilities.getter import get_instance, get_instance_recursively
from theseus.utilities.cuda import get_devices_info, move_to
from theseus.utilities.loading import load_state_dict

from theseus.opt import Config
from theseus.base.pipeline import BasePipeline
from .models import wrapper as refine_model
from theseus.base.models import wrapper as refer_model
from theseus.base.optimizers import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from theseus.semantic.augmentations import TRANSFORM_REGISTRY
from source.semantic2D.losses import LOSS_REGISTRY
from source.semantic2D.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from source.semantic2D.trainer import TRAINER_REGISTRY
from source.semantic2D.metrics import METRIC_REGISTRY
from source.semantic2D.models import MODEL_REGISTRY
from source.semantic2D.callbacks import CALLBACKS_REGISTRY
from theseus.utilities.loggers import LoggerObserver


class Pipeline(BasePipeline):
    """docstring for Pipeline."""

    def __init__(self, opt: Config):
        super(Pipeline, self).__init__(opt)
        self.opt = opt
        self.stage = self.opt["global"]["stage"]
        os.environ["WANDB_ENTITY"] = "kaylode"
        os.environ["WANDB_RUN_GROUP"] = self.stage

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
        self.logger.text("Overidding registry in pipeline...", LoggerObserver.INFO)

    def init_loading(self):

        self.resume = self.opt["global"]["resume"]
        self.pretrained = self.opt["global"]["pretrained"]
        self.last_epoch = -1
        if self.pretrained:
            state_dict = torch.load(self.pretrained)
            if self.stage == "reference":
                if hasattr(self.model.model, 'model'):
                    self.model.model.model = load_state_dict(
                        self.model.model.model, state_dict, "model"
                    )
                else:
                    self.model.model = load_state_dict(
                        self.model.model, state_dict, "model"
                    )
            else:
                state_dict = torch.load(self.pretrained)
                self.model.model.load_network(
                    self.model.model.train_model, state_dict["model"]
                )

        if self.resume:
            state_dict = torch.load(self.resume, map_location=self.device)
            self.optimizer = load_state_dict(self.optimizer, state_dict, "optimizer")
            iters = load_state_dict(None, state_dict, "iters")
            self.last_epoch = iters // len(self.train_dataloader) - 1
            if self.stage == "reference":
                self.model.model.model = load_state_dict(
                    self.model.model.model, state_dict, "model"
                )
            else:
                self.model.model.load_network(
                    self.model.model.train_model, state_dict["model"]
                )

    def init_model(self):
        CLASSNAMES = self.val_dataset.classnames
        model = get_instance(
            self.opt["model"],
            registry=self.model_registry,
            num_classes=len(CLASSNAMES),
            classnames=CLASSNAMES,
            device=self.device,
        )
        if self.stage == "reference":
            model = move_to(model, self.device)
        return model

    def init_metrics(self):
        CLASSNAMES = self.val_dataset.classnames
        self.metrics = get_instance_recursively(
            self.opt["metrics"],
            registry=self.metric_registry,
            num_classes=len(CLASSNAMES),
            classnames=CLASSNAMES,
        )

    def init_model_with_loss(self):
        model = self.init_model()
        criterion = self.init_criterion()
        if self.stage == "reference":
            self.model = refer_model.ModelWithLoss(model, criterion, self.device)
        else:
            self.model = refine_model.ModelWithLoss(model, criterion, self.device)
        self.logger.text(
            f"Number of trainable parameters: {self.model.trainable_parameters():,}",
            level=LoggerObserver.INFO,
        )
        device_info = get_devices_info(self.device_name)
        self.logger.text("Using " + device_info, level=LoggerObserver.INFO)
