from typing import Dict
import time
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
        num_iterations: int=None, 
        time_limit: int = None) -> None:

        self.time_limit = time_limit
        self.num_iterations = num_iterations
        self.target_params = {}
        
    def on_finish(self, logs: Dict=None):
        """
        After the main loop
        """
        for key in self.target_params.keys():
            self.target_params[key] = np.mean(self.target_params[key])

        LOGGER.text("Training Completed!", level=LoggerObserver.INFO)

    def on_train_epoch_start(self, logs: Dict=None):
        """
        Before going to the training loop
        """
        self.time_start = time.time()

    def on_train_batch_end(self, logs:Dict=None):
        """
        On training batch (iteration) end
        """

        loss_dict = logs['loss_dict']

        # Update running loss of batch
        for (key,value) in loss_dict.items():
            if key in self.target_params.keys():
                self.target_params[key].append(value)
            else:
                self.target_params[key] = [value]

        # Early stopping
        if self.time_start:
            if time.time() >= self.time_start:
                raise KeyboardInterrupt("Time limit exceeded")
        
        if self.num_iterations:
            if self.iters > self.num_iterations:
                raise KeyboardInterrupt("Iteration limit exceeded")


    