from typing import Callable, Dict, Optional
import torch
from theseus.opt import Config
from theseus.utilities.getter import (get_instance, get_instance_recursively)
from theseus.utilities.cuda import get_devices_info, move_to
from theseus.utilities.loading import load_state_dict

from theseus.opt import Config
from theseus.base.pipeline import BasePipeline
from theseus.semantic2D.models import wrapper as refine_model
from theseus.base.models import wrapper as refer_model
from theseus.base.optimizers import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from theseus.semantic3D.augmentations import TRANSFORM_REGISTRY
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
        model = get_instance(
            self.opt["model"], 
            registry=self.model_registry, 
            num_classes=len(CLASSNAMES),
            classnames=CLASSNAMES)
        if self.stage == 'reference':
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
            dataset=self.train_dataset,
        )

        self.logger.text(f"Number of training samples: {len(self.train_dataset)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of training iterations each epoch: {len(self.train_dataloader)}", level=LoggerObserver.INFO)

        

        self.val_dataset = get_instance_recursively(
            opt['data']["dataset"]['val'],
            registry=DATASET_REGISTRY,
            transform=self.transform['val'],
        )

        CLASSNAMES = self.val_dataset.classnames

        self.sup_train_dataloader = get_instance(
            opt['data']["dataloader"]['sup_train'],
            registry=DATALOADER_REGISTRY,
            dataset=self.sup_train_dataset,
        )

        self.unsup_train_dataloader1 = get_instance(
            opt['data']["dataloader"]['unsup_train1'],
            registry=DATALOADER_REGISTRY,
            dataset=self.unsup_train_dataset1,
        )

        self.unsup_train_dataloader2 = get_instance(
            opt['data']["dataloader"]['unsup_train2'],
            registry=DATALOADER_REGISTRY,
            dataset=self.unsup_train_dataset2,
        )

        self.val_dataloader = get_instance(
            opt['data']["dataloader"]['val'],
            registry=DATALOADER_REGISTRY,
            dataset=self.val_dataset
        )

        model1 = get_instance(
          self.opt["model1"], 
          registry=MODEL_REGISTRY, 
          classnames=CLASSNAMES,
          num_classes=len(CLASSNAMES)).to(self.device)

        model2 = get_instance(
          self.opt["model2"], 
          registry=MODEL_REGISTRY, 
          classnames=CLASSNAMES,
          num_classes=len(CLASSNAMES)).to(self.device)
          
        sup_criterion = get_instance_recursively(
            self.opt["sup_loss"], 
            registry=LOSS_REGISTRY).to(self.device)
        
        unsup_criterion = get_instance_recursively(
            self.opt["unsup_loss"], 
            registry=LOSS_REGISTRY).to(self.device)

        self.model = ModelWithLoss(
            model1, 
            model2,
            sup_criterion, 
            unsup_criterion, 
            self.device)

        self.metrics = get_instance_recursively(
            self.opt['metrics'], 
            registry=METRIC_REGISTRY, 
            classnames=CLASSNAMES,
            num_classes=len(CLASSNAMES))

        self.optimizer1 = get_instance(
            self.opt["optimizer"],
            registry=OPTIM_REGISTRY,
            params=self.model.model1.parameters(),
        )

        self.optimizer2 = get_instance(
            self.opt["optimizer"],
            registry=OPTIM_REGISTRY,
            params=self.model.model2.parameters(),
        )

        last_epoch = -1
        if self.pretrained:
            state_dict = torch.load(self.pretrained)
            self.model.model1 = load_state_dict(self.model.model1, state_dict, 'model')
            self.model.model2 = load_state_dict(self.model.model2, state_dict, 'model')

        if self.resume:
            state_dict = torch.load(self.resume)
            self.model.model1 = load_state_dict(self.model.model1, state_dict, 'model1')
            self.model.model2 = load_state_dict(self.model.model2, state_dict, 'model2')
            self.optimizer1 = load_state_dict(self.optimizer1, state_dict, 'optimizer1')
            self.optimizer2 = load_state_dict(self.optimizer2, state_dict, 'optimizer2')
            last_epoch = load_state_dict(last_epoch, state_dict, 'epoch')

        self.scheduler1 = get_instance(
            self.opt["scheduler"], registry=SCHEDULER_REGISTRY, optimizer=self.optimizer1,
            **{
                'num_epochs': self.opt["trainer"]['args']['num_epochs'],
                'trainset': self.sup_train_dataset,
                'batch_size': self.opt["data"]['dataloader']['val']['args']['batch_size'],
                'last_epoch': last_epoch,
            }
        )

        self.scheduler2 = get_instance(
            self.opt["scheduler"], registry=SCHEDULER_REGISTRY, optimizer=self.optimizer2,
            **{
                'num_epochs': self.opt["trainer"]['args']['num_epochs'],
                'trainset': self.sup_train_dataset,
                'batch_size': self.opt["data"]['dataloader']['val']['args']['batch_size'],
                'last_epoch': last_epoch,
            }
        )

        self.trainer = get_instance(
            self.opt["trainer"],
            model=self.model,
            suptrainloader=self.sup_train_dataloader,
            unsuptrainloader1=self.unsup_train_dataloader1,
            unsuptrainloader2=self.unsup_train_dataloader2,
            valloader=self.val_dataloader,
            metrics=self.metrics,
            optimizer1=self.optimizer1,
            optimizer2=self.optimizer2,
            scheduler1=self.scheduler1,
            scheduler2=self.scheduler2,
            use_fp16=self.use_fp16,
            save_dir=self.savedir,
            resume=self.resume,
            registry=TRAINER_REGISTRY,
        )

    def infocheck(self):
        self.logger.text(f"Number of trainable parameters: {self.model.trainable_parameters():,}", level=LoggerObserver.INFO)

        device_info = get_devices_info(self.device_name)
        self.logger.text("Using " + device_info, level=LoggerObserver.INFO)

        self.logger.text(f"Number of supervised training samples: {len(self.sup_train_dataset)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of unsupervised training samples: {len(self.unsup_train_dataset1)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of validation samples: {len(self.val_dataset)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of supervised  training iterations each epoch: {len(self.sup_train_dataloader)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of unsupervised1 training iterations each epoch: {len(self.unsup_train_dataloader1)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of unsupervised2 training iterations each epoch: {len(self.unsup_train_dataloader2)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of validation iterations each epoch: {len(self.val_dataloader)}", level=LoggerObserver.INFO)
        self.logger.text(f"Everything will be saved to {self.savedir}", level=LoggerObserver.INFO)

    def initiate(self):
        self.infocheck()

        self.opt.save_yaml(os.path.join(self.savedir, 'pipeline.yaml'))
        self.transform_cfg.save_yaml(os.path.join(self.savedir, 'transform.yaml'))

        tf_logger = TensorboardLogger(self.savedir)
        if self.resume is not None:
            tf_logger.load(find_old_tflog(
                os.path.dirname(os.path.dirname(self.resume))
            ))
        self.logger.subscribe(tf_logger)

        if self.debug:
            self.logger.text("Sanity checking before training...", level=LoggerObserver.DEBUG)
            self.trainer.sanitycheck()


    def fit(self):
        self.initiate()
        self.trainer.fit()

    def evaluate(self):
        self.infocheck()
        writer = ImageWriter(os.path.join(self.savedir, 'samples'))
        self.logger.subscribe(writer)

        self.logger.text("Evaluating...", level=LoggerObserver.INFO)
        self.trainer.evaluate_epoch()