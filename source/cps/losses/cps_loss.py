from typing import Iterable, Dict, Any
import torch
import torch.nn as nn
from source.cps.utils.ramps import sigmoid_rampup

class CPSLoss(nn.Module):
    """Wrapper class for combining multiple loss function 
    
    """
    def __init__(
        self, 
        sup_criterion: nn.Module,
        unsup_criterion: nn.Module,
        consistency: float = 0.1,
        consistency_rampup: float = 200.0,
        **kwargs):
        super().__init__()
        self.unsup_criterion = unsup_criterion
        self.sup_criterion = sup_criterion
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency * sigmoid_rampup(epoch, self.consistency_rampup)

    def forward(self, outputs: Dict, batch: Dict[str, Any], device: torch.device):
        """
        Forward inputs and targets through multiple losses
        """
        epoch = batch['epoch']
        split_pos = batch['split_pos']
        outputs1, outputs2 = outputs['outputs']
        outputs_soft1 = torch.softmax(outputs1, dim=1)
        outputs_soft2 = torch.softmax(outputs2, dim=1)

        pseudo_outputs1 = torch.argmax(
            outputs_soft1[split_pos:].detach(), dim=1, keepdim=False)
        pseudo_outputs2 = torch.argmax(
            outputs_soft2[split_pos:].detach(), dim=1, keepdim=False)

        # Supervised loss
        sup_loss1, sup_loss_dict1 = self.sup_criterion(
            outputs = {'outputs': outputs1[:split_pos]},
            batch = {'targets': batch['targets']},
            device = device
        )
        sup_loss2, sup_loss_dict2 = self.sup_criterion(
            outputs = {'outputs': outputs2[:split_pos]},
            batch = {'targets': batch['targets']},
            device = device
        )

        # Unsupervised loss
        unsup_loss1, unsup_loss_dict1 = self.unsup_criterion(
            outputs = {'outputs': outputs_soft1[split_pos:]},
            batch = {'targets': pseudo_outputs2},
            device = device
        )
        unsup_loss2, unsup_loss_dict2 = self.unsup_criterion(
            outputs = {'outputs': outputs_soft2[split_pos:]},
            batch = {'targets': pseudo_outputs1},
            device = device
        )  

        consistency_weight = self.get_current_consistency_weight(epoch)

        model1_loss = sup_loss1 + consistency_weight * unsup_loss1
        model2_loss = sup_loss2 + consistency_weight * unsup_loss2
        total_loss = model1_loss + model2_loss

        sup_loss_dict1 = {k+'_sup_1': v for k,v in sup_loss_dict1.items()}
        sup_loss_dict2 = {k+'_sup_2': v for k,v in sup_loss_dict2.items()}
        unsup_loss_dict1 = {k+'_unsup_1': v for k,v in unsup_loss_dict1.items()}
        unsup_loss_dict2 = {k+'_unsup_2': v for k,v in unsup_loss_dict2.items()}

        total_loss_dict = {}
        total_loss_dict.update(sup_loss_dict1)
        total_loss_dict.update(sup_loss_dict2)
        total_loss_dict.update(unsup_loss_dict1)
        total_loss_dict.update(unsup_loss_dict2)

        total_loss_dict.update({'Total': total_loss.item()})
        total_loss_dict.update({'weight': consistency_weight})
        return total_loss, total_loss_dict