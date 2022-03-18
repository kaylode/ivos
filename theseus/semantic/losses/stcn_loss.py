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
    def __init__(self):
        super().__init__()
        self.bce = BootstrappedCE()

    def compute(self, data, it):
        loss_dict = defaultdict(int)

        b, num_slices, _, _, _ = data['gt'].shape
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
        loss_dict['T'] = total_loss.item()
        return total_loss, loss_dict