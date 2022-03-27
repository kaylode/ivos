from typing import Any, Dict, Optional
import torch
import numpy as np
from theseus.base.metrics.metric_template import Metric

class mIOU(Metric):
    """ Mean IOU metric for ivos
    """
    def __init__(self, 
            eps: float = 1e-6, 
            num_classes: int = 4,
            ignore_index: Optional[int] = None,
            **kwawrgs):

        self.eps = eps
        self.num_classes=num_classes
        self.ignore_index = ignore_index

        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]): 
        """
        Perform calculation based on prediction and targets
        """
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
        union = A + B - intersect

        return intersect / (union + self.eps)
        
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
        return {"miou" : scores}