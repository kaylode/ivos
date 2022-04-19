from typing import Any, Dict, Optional
import torch
import numpy as np
from collections import OrderedDict
from theseus.base.metrics.metric_template import Metric
from theseus.semantic.metrics.flare22.surface import (
    compute_surface_distances, 
    compute_surface_dice_at_tolerance, 
    compute_dice_coefficient
)

class FLAREMetrics(Metric):
    """ Dice score metric and Normalized Surface Distances for segmentation
    https://github.com/JunMa11/FLARE
    """
    def __init__(self, 
            num_classes: int, 
            thresh: Optional[float] = None,
            **kwawrgs):

        self.thresh = thresh
        self.num_classes = num_classes

        self.reset()

    def update(self, outputs: torch.Tensor, batch: Dict[str, Any]): 
        """
        Perform calculation based on prediction and targets
        """
        # outputs: (batch, num_classes, W, H)
        # targets: (batch, num_classes, W, H)

        targets = batch['gt'].long().squeeze(0).permute(2,1,0)
        info = batch['info']
        spacing = info['case_spacing']
        preds = torch.from_numpy(outputs['out']).long()
        self.compute(targets, preds, spacing)
        self.sample_size += targets.shape[0]
        
    def compute(self, gt_data, seg_data, case_spacing):
        for i in range(1, self.num_classes):
            if np.sum(gt_data==i)==0 and np.sum(seg_data==i)==0:
                DSC_i = 1
                NSD_i = 1
            elif np.sum(gt_data==i)==0 and np.sum(seg_data==i)>0:
                DSC_i = 0
                NSD_i = 0
            else:
                surface_distances = compute_surface_distances(gt_data==i, seg_data==i, case_spacing)
                DSC_i = compute_dice_coefficient(gt_data==i, seg_data==i)
                NSD_i = compute_surface_dice_at_tolerance(surface_distances, 1)
            self.seg_metrics[f'DSC_{i}'].append(DSC_i)
            self.seg_metrics[f'NSD-1mm_{i}'].append(NSD_i)
        
    def reset(self):
        self.seg_metrics = OrderedDict()
        for i in range(1, self.num_classes):
            self.seg_metrics[f'DSC_{i}'] = list()
            self.seg_metrics[f'NSD-1mm_{i}'] = list()
        self.sample_size = 0

    def value(self):
        result_dict = {}
        for i in range(1, self.num_classes):
            result_dict[f'DSC_{i}'] = np.mean(self.seg_metrics[f'DSC_{i}'])
            result_dict[f'NSD-1mm_{i}'] = np.mean(self.seg_metrics[f'NSD-1mm_{i}'])

        result_dict['DSC'] = np.mean([result_dict[f'DSC_{i}'] for i in range(1, self.num_classes)])
        result_dict['NSD-1mm'] = np.mean([result_dict[f'NSD-1mm_{i}'] for i in range(1, self.num_classes)])
        return result_dict