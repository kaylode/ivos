from typing import Dict, Any
import torch
import torch.nn as nn
from theseus.utilities.cuda import move_to
from theseus.semantic.models.stcn.networks.network import STCNTrain

class STCNModel(nn.Module):
    """
    Some simple segmentation models with various pretrained backbones

    name: `str`
        model name [unet, deeplabv3, ...]
    encoder_name : `str` 
        backbone name [efficientnet, resnet, ...]
    num_classes: `int` 
        number of classes
    aux_params: `Dict` 
        auxilliary head
    """
    def __init__(
        self, 
        num_classes: int = 3,
        local_rank: int = 0,
        single_object: bool = False,
        **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.local_rank = local_rank
        self.single_object = single_object

        self.model = nn.parallel.DistributedDataParallel(
            STCNTrain(self.single_object).cuda(), 
            device_ids=[local_rank], 
            output_device=local_rank, 
            broadcast_buffers=False)

    def forward(self, data: Dict):
        # No need to store the gradient outside training
        torch.set_grad_enabled(True)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        Fs = data['rgb'].float()
        Ms = data['gt']
        
        # key features never change, compute once
        k16, kf16_thin, kf16, kf8, kf4 = self.model('encode_key', Fs)

        if self.single_object:
            ref_v = self.model('encode_value', Fs[:,0], kf16[:,0], Ms[:,0])

            # Segment frame 1 with frame 0
            prev_logits, prev_mask = self.model('segment', 
                    k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                    k16[:,:,0:1], ref_v)
            prev_v = self.model('encode_value', Fs[:,1], kf16[:,1], prev_mask)

            values = torch.cat([ref_v, prev_v], 2)

            del ref_v

            # Segment frame 2 with frame 0 and 1
            this_logits, this_mask = self.model('segment', 
                    k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                    k16[:,:,0:2], values)

            out['mask_1'] = prev_mask
            out['mask_2'] = this_mask
            out['logits_1'] = prev_logits
            out['logits_2'] = this_logits
        else:
            sec_Ms = data['sec_gt']
            selector = data['selector']

            ref_v1 = self.model('encode_value', Fs[:,0], kf16[:,0], Ms[:,0], sec_Ms[:,0])
            ref_v2 = self.model('encode_value', Fs[:,0], kf16[:,0], sec_Ms[:,0], Ms[:,0])
            ref_v = torch.stack([ref_v1, ref_v2], 1)

            # Segment frame 1 with frame 0
            prev_logits, prev_mask = self.model('segment', 
                    k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                    k16[:,:,0:1], ref_v, selector)
            
            prev_v1 = self.model('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,0:1], prev_mask[:,1:2])
            prev_v2 = self.model('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,1:2], prev_mask[:,0:1])
            prev_v = torch.stack([prev_v1, prev_v2], 1)
            values = torch.cat([ref_v, prev_v], 3)

            del ref_v

            # Segment frame 2 with frame 0 and 1
            this_logits, this_mask = self.model('segment', 
                    k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                    k16[:,:,0:2], values, selector)

            out['mask_1'] = prev_mask[:,0:1]
            out['mask_2'] = this_mask[:,0:1]
            out['sec_mask_1'] = prev_mask[:,1:2]
            out['sec_mask_2'] = this_mask[:,1:2]

            out['logits_1'] = prev_logits
            out['logits_2'] = this_logits

        return out

        