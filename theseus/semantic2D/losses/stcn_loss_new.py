from typing import Dict, Any, Iterable
import torch
import torch.nn as nn

from collections import defaultdict

class NewSTCNLoss(nn.Module):
    """Wrapper class for combining multiple loss function 
    
    """
    def __init__(self, losses: Iterable[nn.Module], weights=None, **kwargs):
        super().__init__()
        self.losses = losses
        self.weights = [1.0 for _ in range(len(losses))] if weights is None else weights

    def forward(self, outputs: Dict, batch: Dict[str, Any], device: torch.device):
        """
        Forward inputs and targets through multiple losses
        """
        total_loss = 0
        total_loss_dict = defaultdict(float)

        b, num_slices, _, _, _ = batch['targets'].shape
        selector = batch.get('selector', None)

        for weight, loss_fn in zip(self.weights, self.losses):
            
            sum_loss = 0
            sum_loss_dict = defaultdict(float)
            for i in range(1, num_slices):
                # Have to do it in a for-loop like this since not every entry has the second object
                # Well it's not a lot of iterations anyway
                for j in range(b):
                    if selector is not None and selector[j][1] > 0.5:
                        loss, loss_dict = loss_fn(
                            outputs = {'outputs': outputs['logits_%d'%i][j:j+1]},
                            batch = {'targets': batch['cls_gt'][j:j+1,i]},
                            device=device
                        )
                    else:
                        loss, loss_dict = loss_fn(
                            outputs = {'outputs': outputs['logits_%d'%i][j:j+1,:2]},
                            batch = {'targets': batch['cls_gt'][j:j+1,i]},
                            device=device
                        )
                    sum_loss += loss / b

                    for key, value in loss_dict.items():
                        sum_loss_dict[key] += value / b
 
            total_loss += (weight*sum_loss)
            total_loss_dict.update(sum_loss_dict)

        total_loss_dict.update({'Total': total_loss.item()})
        return total_loss, total_loss_dict