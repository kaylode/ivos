from typing import List, Dict
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")

class STCNCallbacks(Callbacks):
    """
    """

    def __init__(self, skip_values, increase_skip_fraction, **kwargs) -> None:
        super().__init__()
        self.skip_values = skip_values
        self.increase_skip_fraction = increase_skip_fraction

    def on_start(self, logs:Dict = None):
        """
        On initialization
        """
        self.num_iterations = self.params['trainer'].num_iterations
        self.increase_skip_epoch = [
            round(self.num_iterations*f) for f in self.increase_skip_fraction]
        LOGGER.text(f"Skip iteration: {self.increase_skip_epoch}", level=LoggerObserver.INFO)

    def on_epoch_start(self, logs:Dict = None):
        """
        On training epoch start
        Calculate current iterations and decide whether to update skip value
        """
        iters = logs['iters']

        if iters!=self.num_iterations and iters>=self.increase_skip_epoch[0]:
            while iters >= self.increase_skip_epoch[0]:
                cur_skip = self.skip_values[0]
                self.skip_values = self.skip_values[1:]
                self.increase_skip_epoch = self.increase_skip_epoch[1:]
            LOGGER.text(f'Increasing skip to: {cur_skip}', level=LoggerObserver.INFO)
            self.renew_loader(cur_skip)

    def on_train_batch_start(self, logs:Dict = None):
        """
        On training batch start
        Save the number of iterations to batch for computing loss 
        """
        iters = logs['iters']
        logs['batch']['iters'] = iters

    def on_val_batch_start(self, logs:Dict = None):
        """
        On validation batch start
        Save the number of iterations to batch for computing loss 
        """
        iters = logs['iters']
        logs['batch']['iters'] = iters

    def renew_loader(self, max_skip: int):
        # //5 because we only have annotation for every five frames
        self.params['trainer'].trainloader.dataset.max_jump = max_skip
        LOGGER.text(f'Renewed with skip: {max_skip}', level=LoggerObserver.INFO)