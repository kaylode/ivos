from typing import Any, Dict, Optional
from theseus.base.metrics.metric_template import Metric

def compute_tensor_iu(seg, gt):
    intersection = (seg & gt).float().sum()
    union = (seg | gt).float().sum()

    return intersection, union

def compute_tensor_iou(seg, gt):
    intersection, union = compute_tensor_iu(seg, gt)
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou 

class mIOU(Metric):
    """ Mean IOU metric for ivos
    thresh: `float`
        threshold for binary segmentation
    """
    def __init__(self, 
            eps: float = 1e-6, 
            thresh: Optional[float] = 0.5,
            **kwawrgs):

        self.thresh = thresh
        self.eps = eps

        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]): 
        """
        Perform calculation based on prediction and targets
        """
        b, num_slices, _, _, _ = batch['gt'].shape
        selector = batch.get('selector', None)

        for i in range(1, num_slices):
            iou = compute_tensor_iou(
                outputs['mask_%d'%i]>self.thresh, batch['gt'][:,i]>self.thresh)
            self.tar_iou += iou

            if selector is not None:
                sec_iou = compute_tensor_iou(
                    outputs['sec_mask_%d'%i]>self.thresh, batch['sec_gt'][:,i]>self.thresh)
                self.sec_iou += sec_iou

        self.sample_size += num_slices-1 #exclude first frame
        
    def reset(self):
        self.tar_iou = 0
        self.sec_iou = 0
        self.sample_size = 0

    def value(self):
        tar_iou_score = self.tar_iou / self.sample_size #mean over number of samples
        sec_iou_score = self.sec_iou / self.sample_size #mean over number of samples
        
        return {
            "iou1" : tar_iou_score,
            "iou2": sec_iou_score,
            "miou": (tar_iou_score+sec_iou_score) / 2
        }