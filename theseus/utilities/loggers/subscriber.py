from typing import Callable, Dict, Optional


class LoggerSubscriber:
    def __init__(self, **kwargs) -> None:
        pass

    def log_scalar(self, **kwargs):
        return

    def log_figure(self, **kwargs):
        return

    def log_torch_module(self, **kwargs):
        return

    def log_text(self, **kwargs):
        return

    def log_embedding(self, **kwargs):
        return

    def log_spec_text(self, **kwargs):
        return

    def log_table(self, **kwargs):
        return