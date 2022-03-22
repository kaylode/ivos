from typing import Any, Dict, Optional
import torch
import numpy as np
from theseus.base.metrics.metric_template import Metric

class DiceScore(Metric):
    """ Dice score metric for segmentation
    num_classes: `int`
        number of classes 
    eps: `float`
        epsilon to avoid zero division
    thresh: `float`
        threshold for binary segmentation
    """
    def __init__(self, 
            num_classes: int, 
            smooth: int = 1, 
            thresh: Optional[float] = None,
            ignore_index: Optional[int] = None,
            **kwawrgs):

        self.thresh = thresh
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

        self.reset()

    def update(self, outputs: torch.Tensor, batch: Dict[str, Any]): 
        """
        Perform calculation based on prediction and targets
        """
        # outputs: (batch, num_classes, W, H)
        # targets: (batch, num_classes, W, H)

        targets = batch['gt'].long().squeeze(0).permute(2,1,0)
        preds = torch.from_numpy(outputs['out']).long()

        one_hot_predicts = torch.nn.functional.one_hot(
              preds.long(), 
              num_classes=self.num_classes).permute(0, 3, 1, 2)

        one_hot_targets = torch.nn.functional.one_hot(
              targets.long(), 
              num_classes=self.num_classes).permute(0, 3, 1, 2)

        for cl in range(self.num_classes):
            cl_pred = one_hot_predicts[:,cl,:,:]
            cl_target = one_hot_targets[:,cl,:,:]
            score = self.binary_compute(cl_pred, cl_target)
            self.scores_list[cl] += sum(score)

        self.sample_size += targets.shape[0]
        

    def binary_compute(self, predict: torch.Tensor, target: torch.Tensor):
        # outputs: (batch, W, H)
        # targets: (batch, W, H)

        intersect = torch.sum(target*predict, dim=(-1, -2))
        A = torch.sum(target, dim=(-1, -2))
        B = torch.sum(predict, dim=(-1, -2))
        union = A + B
        return (2. * intersect + self.smooth)  / (union + self.smooth)
        
    def reset(self):
        self.scores_list = np.zeros(self.num_classes)
        self.sample_size = 0

    def value(self):
        scores_each_class = self.scores_list / self.sample_size #mean over number of samples
        if self.ignore_index is not None:
            scores_each_class[self.ignore_index] = 0
            scores = sum(scores_each_class) / (self.num_classes - 1)
        else:
            scores = sum(scores_each_class) / self.num_classes
        return {"dice" : scores}