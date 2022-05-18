import torch
from torch import nn
from theseus.utilities.cuda import move_to

class ModelWithLoss(nn.Module):
    """Add utilitarian functions for module to work with pipeline

    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat

    """

    def __init__(self, model: nn.Module, criterion: nn.Module, device: torch.device):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.device = device

    def forward(self, batch, metrics=None):
        """
        Forward the batch through models, losses and metrics
        If some parameters are needed, it's best to include in the batch
        """
        outputs = self.model(batch, self.device)

        if metrics is not None:
            for metric in metrics:
                metric.update(outputs, batch)
            return {
                'model_outputs': outputs,
                'loss': 0,
                'loss_dict': {"None": 0}
            }
        else:
            loss, loss_dict = self.criterion(outputs, batch, self.device)
        
            return {
                'model_outputs': outputs,
                'loss': loss,
                'loss_dict': loss_dict
            }

    def training_step(self, batch):
        self.model.train_mode()
        return self.forward(batch)

    def evaluate_step(self, batch, metrics=None):
        self.model.eval_mode()
        return self.forward(batch, metrics)

    def state_dict(self):
        return self.model.state_dict()

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)