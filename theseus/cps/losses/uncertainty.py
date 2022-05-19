from typing import Dict, List, Any
import torch
from torch import nn
from theseus.utilities.cuda import move_to
from theseus.cps.utils.ramps import sigmoid_rampup

class UELoss(nn.Module):
    r"""
    UELoss is warper of cross-entropy loss
    https://arxiv.org/pdf/2003.03773v3.pdf
    https://github.com/layumi/Seg-Uncertainty
    """

    def __init__(self, lambda_me_target: float = 0, lambda_kl_target: float = 0, weights=None, **kwargs):
        super(UELoss, self).__init__()
        self.kl_distance = nn.KLDivLoss( reduction = 'none')
        self.sm = torch.nn.Softmax(dim = 1)
        self.log_sm = torch.nn.LogSoftmax(dim = 1)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1.0, 1.0]
        self.lambda_kl_target = lambda_kl_target
        self.lambda_me_target = lambda_me_target

    def estimate(self, loss, tensor1, tensor2):
        variance = torch.sum(self.kl_distance(self.log_sm(tensor1),self.sm(tensor2)), dim=1) 
        exp_variance = torch.exp(-variance)
        loss = torch.mean(loss*exp_variance) + torch.mean(variance)
        return loss, variance

    def forward(self, outputs: Dict, device: torch.device):
        outputs1, outputs2 = outputs['outputs']
        obj_loss1, obj_loss2 = outputs['obj_losses']
        # total_loss = (
        #     self.weights[0] * self.estimate(obj_loss1, outputs1, outputs2) 
        #     + self.weights[1] * self.estimate(obj_loss2, outputs2, outputs1)
        # )

        loss = 0
        loss_me = 0.0
        if self.lambda_me_target>0:
            confidence_map = torch.sum( self.sm(outputs1 + outputs2)**2, 1).detach()
            loss_me = -torch.mean( 
                confidence_map * torch.sum( 
                    self.sm(outputs1 + outputs2) * self.log_sm(outputs1 + outputs2), 1) 
                )
            loss += self.lambda_me_target * loss_me

        loss_kl = 0.0
        if self.lambda_kl_target>0:
            n, c, h, w = outputs1.shape
            with torch.no_grad():
                mean_pred = self.sm(outputs1 + outputs2) 
            loss_kl = ( 
                self.kl_distance(self.log_sm(outputs2) , mean_pred)  
                + self.kl_distance(self.log_sm(outputs1) , mean_pred)
            )/(n*h*w)
            loss += self.lambda_kl_target * loss_kl

        loss_dict = {
            'UE': loss.item(),
        }

        return loss, loss_dict

class UncertaintyCPSLoss(nn.Module):
    """Wrapper class for combining multiple loss function 
    
    """
    def __init__(
        self, 
        sup_criterion: nn.Module,
        unsup_criterion: nn.Module,
        consistency: float = 0.1,
        consistency_rampup: float = 200.0,
        weights: List[float] = [1.0, 1.0],
        **kwargs):
        super().__init__()
        self.unsup_criterion = unsup_criterion
        self.sup_criterion = sup_criterion
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup
        self.uncertainty_estimation = UELoss(weights=weights)

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

        uncertainty_loss, uncertainty_dict = self.uncertainty_estimation(
            outputs = {
                "outputs": [outputs1, outputs2],
                'obj_losses': [unsup_loss1, unsup_loss2]
            },
            device = device
        )

        consistency_weight = self.get_current_consistency_weight(epoch)

        model1_loss = sup_loss1 #+ consistency_weight * unsup_loss1
        model2_loss = sup_loss2 #+ consistency_weight * unsup_loss2
        total_loss = model1_loss + model2_loss + uncertainty_loss

        sup_loss_dict1 = {k+'_sup_1': v for k,v in sup_loss_dict1.items()}
        sup_loss_dict2 = {k+'_sup_2': v for k,v in sup_loss_dict2.items()}
        unsup_loss_dict1 = {k+'_unsup_1': v for k,v in unsup_loss_dict1.items()}
        unsup_loss_dict2 = {k+'_unsup_2': v for k,v in unsup_loss_dict2.items()}

        total_loss_dict = {}
        total_loss_dict.update(sup_loss_dict1)
        total_loss_dict.update(sup_loss_dict2)
        total_loss_dict.update(unsup_loss_dict1)
        total_loss_dict.update(unsup_loss_dict2)
        total_loss_dict.update(uncertainty_dict)

        total_loss_dict.update({'Total': total_loss.item()})
        total_loss_dict.update({'weight': consistency_weight})
        return total_loss, total_loss_dict