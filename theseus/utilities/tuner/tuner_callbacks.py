from typing import Dict
import time
import optuna
import numpy as np
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class TuningCallbacks(Callbacks):
    """
    min_lr: minimum learning rate to investigate
    max_lr: maximum learning rate to investigate
    num_training: number of learning rates to test
    mode: Search strategy to update learning rate after each batch:
        - ``'exponential'`` (default): Will increase the learning rate exponentially.
        - ``'linear'``: Will increase the learning rate linearly.
    early_stop_threshold: threshold for stopping the search. If the
        loss at any point is larger than early_stop_threshold*best_loss
        then the search is stopped. To disable, set to None.
    """
    def __init__(
        self, 
        trial: optuna.trial.Trial,
        num_iterations: int=None, 
        time_limit: int = None) -> None:

        super().__init__()
        self.time_limit = time_limit
        self.num_iterations = num_iterations
        self.trial = trial
        self.target_params = {}
    
    def on_start(self, logs: Dict=None):
        """
        Before going to the main loop
        """
        LOGGER.text(f'Starting trial {self.trial.number}...', level=LoggerObserver.INFO)
        LOGGER.text(f"Trial parameters: {self.trial.params}", level=LoggerObserver.INFO)
        self.time_start = time.time()

    def on_finish(self, logs: Dict=None):
        """
        After the main loop
        """
        for key in self.target_params.keys():
            self.target_params[key] = np.mean(self.target_params[key])

        LOGGER.text(f"Trial {self.trial.number} finished", level=LoggerObserver.INFO)

    def on_train_batch_end(self, logs:Dict=None):
        """
        On training batch (iteration) end
        """

        iters = logs['iters']
        loss_dict = logs['loss_dict']
        trainer = self.params['trainer']

        # Update running loss of batch
        for (key,value) in loss_dict.items():
            key  = 'train_' + key
            if key in self.target_params.keys():
                self.target_params[key].append(value)
            else:
                self.target_params[key] = [value]

        # Early stopping
        if self.time_start:
            LOGGER.text(f"Time passed: {time.time() - self.time_start}", level=LoggerObserver.INFO)
            if time.time() - self.time_start >= self.time_limit:
                trainer.shutdown_training = True
                LOGGER.text("Time limit exceeded", level=LoggerObserver.DEBUG)
        
        if self.num_iterations:
            LOGGER.text(f"Number of iterations passed: {iters}", level=LoggerObserver.INFO)
            if iters > self.num_iterations:
                trainer.shutdown_training = True
                LOGGER.text("Iteration limit exceeded", level=LoggerObserver.DEBUG)
        
        # Restart the clock for validation
        if trainer.shutdown_training:
            self.time_start = time.time()


    def on_val_batch_end(self, logs:Dict=None):
        """
        On validation batch (iteration) end
        """

        iters = logs['iters']
        loss_dict = logs['loss_dict']
        trainer = self.params['trainer']

        # Update running loss of batch
        for (key,value) in loss_dict.items():
            key  = 'val_' + key
            if key in self.target_params.keys():
                self.target_params[key].append(value)
            else:
                self.target_params[key] = [value]

        # Early stopping
        if self.time_start:
            LOGGER.text(f"Time passed: {time.time() - self.time_start}", level=LoggerObserver.INFO)
            if time.time() - self.time_start >= self.time_limit:
                trainer.shutdown_validation = True
                trainer.shutdown_all = True
                LOGGER.text("Time limit exceeded", level=LoggerObserver.DEBUG)
        
        if self.num_iterations:
            LOGGER.text(f"Number of iterations passed: {iters}", level=LoggerObserver.INFO)
            if iters > self.num_iterations:
                trainer.shutdown_validation = True
                trainer.shutdown_all = True
                LOGGER.text("Iteration limit exceeded", level=LoggerObserver.DEBUG)

    def on_val_epoch_end(self, logs:Dict=None):
        metric_dict = logs['metric_dict']
        # Update running metric of batch
        for (key,value) in metric_dict.items():
            key  = 'val_' + key
            if key in self.target_params.keys():
                self.target_params[key].append(value)
            else:
                self.target_params[key] = [value]