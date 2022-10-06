from typing import Any, Dict, Optional
import torch
import numpy as np
from collections import OrderedDict
from theseus.base.metrics.metric_template import Metric
from source.semantic2D.metrics.nsd.surface import (
    compute_surface_distances, 
    compute_surface_dice_at_tolerance, 
)

class NormalizedSurfaceDistance(Metric):
    """ Dice score metric and Normalized Surface Distances for segmentation
    https://github.com/JunMa11/FLARE
    """
    def __init__(self, 
            num_classes: int, 
            calc_each_class: bool = False,
            **kwawrgs):

        self.num_classes = num_classes
        self.calc_each_class = calc_each_class
        self.reset()

    def update(self, outputs: torch.Tensor, batch: Dict[str, Any]): 
        """
        Perform calculation based on prediction and targets
        """
        # outputs: (batch, num_classes, W, H)
        # targets: (batch, num_classes, W, H)

        targets = batch['targets'].long().squeeze(0).permute(2,1,0).numpy()
        preds = outputs['outputs'].astype(int)
        spacing = batch['info']['case_spacing']
        self.compute(targets, preds, spacing)
        self.sample_size += 1
        
    def compute(self, gt_data, seg_data, case_spacing):
        for i in range(1, self.num_classes):
            if np.sum(gt_data==i)==0 and np.sum(seg_data==i)==0:
                NSD_i = 1
            elif np.sum(gt_data==i)==0 and np.sum(seg_data==i)>0:
                NSD_i = 0
            else:
                surface_distances = compute_surface_distances(gt_data==i, seg_data==i, case_spacing)
                NSD_i = compute_surface_dice_at_tolerance(surface_distances, 1)
            self.seg_metrics[f'NSD-1mm_{i}'].append(NSD_i)
        
    def reset(self):
        self.seg_metrics = OrderedDict()
        for i in range(1, self.num_classes):
            self.seg_metrics[f'NSD-1mm_{i}'] = list()
        self.sample_size = 0

    def value(self):
        result_dict = {}
        for i in range(1, self.num_classes):
            result_dict[f'NSD-1mm_{i}'] = np.mean(self.seg_metrics[f'NSD-1mm_{i}'])
        result_dict['NSD-1mm-avg'] = np.mean([result_dict[f'NSD-1mm_{i}'] for i in range(1, self.num_classes)])

        if self.calc_each_class:
            return result_dict
        else:
            return {
                'NSD-1mm': result_dict['NSD-1mm-avg']
            }