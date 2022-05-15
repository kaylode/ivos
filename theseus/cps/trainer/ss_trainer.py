import time
from tqdm import tqdm
import torch
from torch.cuda import amp

from theseus.base.trainer.base_trainer import BaseTrainer

from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class SemiSupervisedTrainer(BaseTrainer):
    """Trainer for supervised tasks
    
    model : `torch.nn.Module`
        Wrapper model with loss 
    trainloader : `torch.utils.DataLoader`
        DataLoader for training
    valloader : `torch.utils.DataLoader`
        DataLoader for validation
    metrics: `List[Metric]`
        list of metrics for evaluation
    optimizer: `torch.optim.Optimizer`
        optimizer for parameters update
    scheduler: `torch.optim.lr_scheduler.Scheduler`
        learning rate schedulers

    """
    def __init__(
        self, 
        model, 
        trainloader, 
        valloader,
        metrics,
        optimizers,
        schedulers,
        **kwargs):

        super().__init__(**kwargs)

        self.model = model
        self.metrics = metrics 
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.trainloader = trainloader
        self.valloader = valloader
        self.use_cuda = next(self.model.parameters()).is_cuda

        if len(self.schedulers):
            self.step_per_epochs = [sched.step_per_epoch for sched in self.schedulers]

        # Flags for shutting down training or validation stages
        self.shutdown_training = False
        self.shutdown_validation = False


    def training_epoch(self):
        """
        Perform training one epoch
        """
        self.model.train()
        self.callbacks.run('on_train_epoch_start')
        self.optimizer.zero_grad()
        for i, batch in enumerate(self.trainloader):

            # Check if shutdown flag has been turned on
            if self.shutdown_training or self.shutdown_all:
                break

            self.callbacks.run('on_train_batch_start', {
                'batch': batch,
                'iters': self.iters,
                'num_iterations': self.num_iterations
            })

            # Gradient scaler
            with amp.autocast(enabled=self.use_amp):
                outputs = self.model.training_step(batch)
                loss = outputs['loss']
                loss_dict = outputs['loss_dict']

            # Backward loss
            for optimizer in self.optimizers:
                self.scaler(loss, optimizer)
            self.scaler.step(optimizer[0], clip_grad=self.clip_grad, parameters=self.model.model1.parameters())
            self.scaler.step(optimizer[1], clip_grad=self.clip_grad, parameters=self.model.model2.parameters())
            
            # Optmizer step
            for sched, step_per_epoch in zip(self.schedulers, self.step_per_epochs):
                if sched is not None and not step_per_epoch:
                    sched.step()

            for optimizer in self.optimizers:
                optimizer.zero_grad()

            if self.use_cuda:
                torch.cuda.synchronize()

            # Calculate current iteration
            self.iters = self.iters + 1

            # Get learning rate
            lrl = [x['lr'] for x in self.optimizers[0].param_groups]
            lr = sum(lrl) / len(lrl)

            self.callbacks.run('on_train_batch_end', {
                'loss_dict': loss_dict,
                'iters': self.iters,
                'num_iterations': self.num_iterations,
                'lr': lr
            })

        for sched, step_per_epoch in zip(self.schedulers, self.step_per_epochs):
            if sched is not None and not step_per_epoch:
                sched.step()

        self.callbacks.run('on_train_epoch_end', {
            'last_batch': batch,
            'iters': self.iters
        })
        

    @torch.no_grad()   
    def evaluate_epoch(self):
        """
        Perform validation one epoch
        """
        self.model.eval()

        self.callbacks.run('on_val_epoch_start')
        for batch in tqdm(self.valloader):

            # Check if shutdown flag has been turned on
            if self.shutdown_validation or self.shutdown_all:
                break

            self.callbacks.run('on_val_batch_start', {
                'batch': batch,
                'iters': self.iters,
                'num_iterations': self.num_iterations
            })

            # Gradient scaler
            with amp.autocast(enabled=self.use_amp):
                outputs = self.model.evaluate_step(batch, self.metrics)
                loss_dict = outputs['loss_dict']

            self.callbacks.run('on_val_batch_end', {
                'loss_dict': loss_dict,
                'iters': self.iters,
                'num_iterations': self.num_iterations,
            })
                
        metric_dict = {}
        for metric in self.metrics:
            metric_dict.update(metric.value())
            metric.reset()  

        self.callbacks.run("on_val_epoch_end", {
            'metric_dict': metric_dict,
            'iters': self.iters,
            'num_iterations': self.num_iterations,
            'last_batch': batch,
            'last_outputs': outputs['model_outputs']
        })
