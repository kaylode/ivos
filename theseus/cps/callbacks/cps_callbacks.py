from typing import List, Dict
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")

class CPSCallbacks(Callbacks):
    """
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def on_start(self, logs:Dict = None):
        """
        On initialization
        """
        train_dataloader = self.params['trainer'].trainloader
        self.iter_length = len(train_dataloader)

    def on_train_batch_start(self, logs:Dict = None):
        """
        On training batch start
        Save the number of iterations to batch for computing loss 
        """
        iters = logs['iters']
        logs['batch']['epoch'] = iters // self.iter_length

    def on_val_batch_start(self, logs:Dict = None):
        """
        On validation batch start
        Save the number of iterations to batch for computing loss 
        """
        iters = logs['iters']
        logs['batch']['epoch'] = iters // self.iter_length