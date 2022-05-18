from typing import Callable, Dict, Optional
import torch
from theseus.opt import Config
from theseus.utilities.getter import (get_instance, get_instance_recursively)
from theseus.utilities.cuda import get_devices_info, move_to
from theseus.utilities.loading import load_state_dict

from theseus.opt import Config
from theseus.base.pipeline import BasePipeline
from theseus.cps.models import wrapper as refer_model
from theseus.base.optimizers import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from theseus.semantic3D.augmentations import TRANSFORM_REGISTRY
from theseus.cps.losses import LOSS_REGISTRY
from theseus.cps.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.cps.trainer import TRAINER_REGISTRY
from theseus.semantic2D.metrics import METRIC_REGISTRY
from theseus.cps.models import MODEL_REGISTRY
from theseus.cps.callbacks import CALLBACKS_REGISTRY
from theseus.utilities.loggers import LoggerObserver



class Pipeline(BasePipeline):
    """docstring for Pipeline."""

    def __init__(
        self,
        opt: Config
    ):
        super(Pipeline, self).__init__(opt)
        self.opt = opt
        self.stage = self.opt['global']['stage']

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

    def init_model(self):
        CLASSNAMES = self.val_dataset.classnames
        model = get_instance_recursively(
            self.opt["model"], 
            registry=self.model_registry, 
            num_classes=len(CLASSNAMES),
            classnames=CLASSNAMES)
        model = move_to(model, self.device)
        return model
    

    def init_train_dataloader(self):
        # DataLoaders
        self.transform = get_instance_recursively(
            self.transform_cfg, registry=self.transform_registry
        )
        self.sup_train_dataset = get_instance_recursively(
            self.opt['data']["dataset"]['train'],
            registry=self.dataset_registry,
            transform=self.transform['sup_train'],
        )

        self.unsup_train_dataset = get_instance_recursively(
            self.opt['data']["dataset"]['unsup_train'],
            registry=self.dataset_registry,
            transform=self.transform['unsup_train'],
        )

        self.train_dataloader = get_instance_recursively(
            self.opt['data']["dataloader"]['train'],
            registry=self.dataloader_registry,
            dataset_l=self.sup_train_dataset,
            dataset_u=self.unsup_train_dataset,
        )

        self.logger.text(f"Number of labeled training samples: {len(self.sup_train_dataset)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of unlabeled training samples: {len(self.unsup_train_dataset)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of training iterations each epoch: {len(self.train_dataloader)}", level=LoggerObserver.INFO)

    def init_optimizer(self):
        self.optimizers = [
            get_instance(
                self.opt["optimizer"],
                registry=self.optimizer_registry,
                params=self.model.model.model1.parameters(),
            ),
            get_instance(
                self.opt["optimizer"],
                registry=self.optimizer_registry,
                params=self.model.model.model2.parameters(),
            ),
        ]

    def init_loading(self):
        self.resume = self.opt['global']['resume']
        self.pretrained = self.opt['global']['pretrained']
        self.last_epoch = -1
        # if self.pretrained:
        #     state_dict = torch.load(self.pretrained)
        #     self.model.model = load_state_dict(self.model.model, state_dict, 'model')

        # if self.resume:
        #     state_dict = torch.load(self.resume)
        #     self.model.model = load_state_dict(self.model.model, state_dict, 'model')
        #     self.optimizer = load_state_dict(self.optimizer, state_dict, 'optimizer')
        #     iters = load_state_dict(None, state_dict, 'iters')
        #     self.last_epoch = iters//len(self.train_dataloader) - 1


    def init_scheduler(self):
        self.schedulers = [
            get_instance(
                self.opt["scheduler"], registry=self.scheduler_registry, optimizer=self.optimizers[0],
                **{
                    'num_epochs': self.opt["trainer"]['args']['num_iterations'] // len(self.train_dataloader),
                    'trainset': self.sup_train_dataset,
                    'batch_size': self.opt["data"]['dataloader']['val']['args']['batch_size'],
                    'last_epoch': self.last_epoch,
                }
            ),
            get_instance(self.opt["scheduler"], registry=self.scheduler_registry, optimizer=self.optimizers[1],
                **{
                    'num_epochs': self.opt["trainer"]['args']['num_iterations'] // len(self.train_dataloader),
                    'trainset': self.sup_train_dataset,
                    'batch_size': self.opt["data"]['dataloader']['val']['args']['batch_size'],
                    'last_epoch': self.last_epoch,
                }
            )
        ]

        # if self.resume:
        #     state_dict = torch.load(self.resume)
        #     self.scheduler = load_state_dict(self.scheduler, state_dict, 'scheduler')

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
        self.model = refer_model.ModelWithLoss(model, criterion, self.device)
        self.logger.text(f"Number of trainable parameters: {self.model.trainable_parameters():,}", level=LoggerObserver.INFO)
        device_info = get_devices_info(self.device_name)
        self.logger.text("Using " + device_info, level=LoggerObserver.INFO)
            
    def init_trainer(self, callbacks):
        self.trainer = get_instance(
            self.opt["trainer"],
            model=self.model,
            trainloader=getattr(self, "train_dataloader", None),
            valloader=getattr(self, "val_dataloader", None),
            metrics=getattr(self, "metrics", None),
            optimizers=getattr(self, "optimizers", None),
            schedulers=getattr(self, "schedulers", None),
            debug=self.debug,
            registry=self.trainer_registry,
            callbacks=callbacks
        )