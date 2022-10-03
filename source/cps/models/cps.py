import torch
from torch import nn
from typing import List, Dict, Any

class CrossPseudoSupervision(nn.Module):
    """Add utilitarian functions for module to work with pipeline
    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat
    """

    def __init__(
        self, 
        model1: nn.Module, 
        model2: nn.Module, 
        reduction: str = 'sum',
        **kwargs):

        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.num_classes = self.model1.num_classes
        self.training = True
        self.reduction = reduction
    
    def get_model(self):
        return self

    def ensemble_learning(self, logit1, logit2, reduction='max'):
        prob1 = torch.softmax(logit1, dim=1)
        prob2 = torch.softmax(logit2, dim=1)

        output = torch.stack([prob1, prob2], dim=0) # [2, B, C, H, W]
        if reduction == 'sum':
            output = output.sum(dim=0) #[B, C, H, W]
        elif reduction == 'max':
            output, _ = output.max(dim=0) #[B, C, H, W]
        elif reduction == 'first':
            return logit1
        elif reduction == 'second':
            return logit2

        return output

    def eval_mode(self):
        """
        Switch to eval mode
        """
        self.training = False

    def train_mode(self):
        """
        Switch to train mode
        """
        self.training = True

    def forward(self, batch: Dict, device: torch.device):
        outputs1 = self.model1(batch, device)['outputs']
        outputs2 = self.model2(batch, device)['outputs']
        if self.training:
            return {
                'outputs': [outputs1, outputs2],
            }
        else:
            outputs = self.ensemble_learning(outputs1, outputs2, reduction=self.reduction)
            return {
                'outputs': outputs
            }
    
    @torch.no_grad()
    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        outputs1 = self.model1(adict, device)['outputs']
        outputs2 = self.model2(adict, device)['outputs']

        probs = self.ensemble_learning(outputs1, outputs2, reduction=self.reduction)
        probs = torch.softmax(probs, dim=1) # B, C, H, W

        if 'weights' in adict.keys():
            weights = adict['weights'] # C
            for i, weight in enumerate(weights):
                probs[:, i] *= weight

        if 'thresh' in adict.keys():
            thresh = adict['thresh']
            probs = (probs > thresh).float()

        predict = torch.argmax(probs, dim=1)

        predict = predict.detach().cpu().squeeze().numpy()
        return {
            'masks': predict
        } 