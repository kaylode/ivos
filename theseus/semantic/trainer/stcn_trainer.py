import torch
from torch.cuda import amp

from theseus.base.trainer.supervised_trainer import SupervisedTrainer

from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class STCNTrainer(SupervisedTrainer):
    """Trainer for stcn tasks


    """
    def __init__(
        self, 
        **kwargs):

        super().__init__(**kwargs)

    def training_epoch(self):
        """
        Perform training one epoch
        """
        self.model.train()
        self.callbacks.run('on_train_epoch_start')
        self.optimizer.zero_grad()
        for i, batch in enumerate(self.trainloader):
            self.callbacks.run('on_train_batch_start', {'batch': batch})

            # Gradient scaler
            with amp.autocast(enabled=self.use_amp):
                outputs = self.model.training_step(batch)
                loss = outputs['loss']
                loss_dict = outputs['loss_dict']

                # Backward loss
                self.scaler(loss, self.optimizer)
                
                # Optmizer step
                self.scaler.step(self.optimizer)
                if not self.step_per_epoch:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            if self.use_cuda:
                torch.cuda.synchronize()

            # Calculate current iteration
            self.iters = self.iters + 1

            # Get learning rate
            lrl = [x['lr'] for x in self.optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            self.callbacks.run('on_train_batch_end', {
                'loss_dict': loss_dict,
                'iters': self.iters,
                'num_iterations': self.num_iterations,
                'lr': lr
            })

        if self.step_per_epoch:
            self.scheduler.step()

        self.callbacks.run('on_train_epoch_end', {
            'last_batch': batch,
            'iters': self.iters
        })