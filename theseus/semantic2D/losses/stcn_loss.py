from typing import Dict, Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class STCNLoss:
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15, **kwargs):
        super().__init__()
        self.bce = BootstrappedCE(start_warm=start_warm, end_warm=end_warm, top_p=top_p)

    def compute(self, data, it):
        loss_dict = defaultdict(int)

        b, num_slices, _, _, _ = data['targets'].shape
        selector = data.get('selector', None)

        total_loss = 0

        for i in range(1, num_slices):
            # Have to do it in a for-loop like this since not every entry has the second object
            # Well it's not a lot of iterations anyway
            for j in range(b):
                if selector is not None and selector[j][1] > 0.5:
                    loss, p = self.bce(data['logits_%d'%i][j:j+1], data['cls_gt'][j:j+1,i], it)
                else:
                    loss, p = self.bce(data['logits_%d'%i][j:j+1,:2], data['cls_gt'][j:j+1,i], it)

                loss_dict['loss_%d'%i] += loss / b
                loss_dict['p'] += p / b / (num_slices-1)

            total_loss += loss_dict['loss_%d'%i]

        loss_dict = {k:v.item() for k, v in loss_dict.items() if isinstance(v, torch.Tensor)}
        loss_dict['ohemCE'] = total_loss.item()
        return total_loss, loss_dict

class STCNLossV2(nn.Module):
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