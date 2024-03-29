"""
eval_network.py - Evaluation version of the network
The logic is basically the same
but with top-k and some implementation optimization

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from source.semantic2D.models.stcn.backbone.modules import (
    KeyEncoder, ValueEncoder, KeyProjection
)

from source.semantic2D.models.stcn.networks.network import Decoder


class STCNEval(nn.Module):
    def __init__(self, 
        key_backbone:str = 'resnet50', 
        value_backbone:str = 'resnet18-mod',
        pretrained: bool = True):
        super().__init__()
        self.key_encoder = KeyEncoder(key_backbone, pretrained) 
        f16_dim = self.key_encoder.model.f16_dim #1024
        f8_dim = self.key_encoder.model.f8_dim #512
        f4_dim = self.key_encoder.model.f4_dim #256
        
        self.value_encoder = ValueEncoder(value_backbone, pretrained, key_dim=f16_dim, out_dim=f8_dim) 

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(f16_dim, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(f16_dim, f8_dim, kernel_size=3, padding=1)

        self.decoder = Decoder(f16_dim=f16_dim, f8_dim=f8_dim, f4_dim=f4_dim)

    def encode_value(self, frame, kf16, masks): 
        k, _, h, w = masks.shape
        num_channels = frame.shape[1]

        # Extract memory key/value for a frame with multiple masks
        frame = frame.view(1, num_channels, h, w).repeat(k, 1, 1, 1)
        # Compute the "others" mask
        if k != 1:
            others = torch.cat([
                torch.sum(
                    masks[[j for j in range(k) if i!=j]]
                , dim=0, keepdim=True)
            for i in range(k)], 0)
        else:
            others = torch.zeros_like(masks)

        f16 = self.value_encoder(frame, kf16.repeat(k,1,1,1), masks, others)

        return f16.unsqueeze(2)

    def encode_key(self, frame):
        f16, f8, f4 = self.key_encoder(frame)
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)
        return k16, f16_thin, f16, f8, f4

    def segment_with_query(self, mem_bank, qf8, qf4, qk16, qv16): 
        k = mem_bank.num_objects
        readout_mem = mem_bank.match_memory(qk16)
        qv16 = qv16.expand(k, -1, -1, -1)
        qv16 = torch.cat([readout_mem, qv16], 1)
        
        return torch.sigmoid(self.decoder(qv16, qf8, qf4))