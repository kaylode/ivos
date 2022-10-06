# coding=utf-8
# This file borrowed from Swin-UNet: https://github.com/HuCaoFighting/Swin-Unet

from typing import Dict, Any
import torch
import torch.nn as nn
import copy
from theseus.utilities.loading import load_state_dict
from theseus.utilities.cuda import move_to, detach
from source.semantic2D.models.swin.swin_module import SwinTransformerSys
from source.semantic2D.models.swin.config import get_config, load_pretrained_model

class SwinUnet(nn.Module):
    """
    Source: https://github.com/HuCaoFighting/Swin-Unet
    """
    def __init__(
        self, 
        model_name: str, 
        num_classes: int, 
        img_size: int = 512, 
        in_channels: int = 3,
        pretrained: bool = True,
        **kwargs):

        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.config = get_config(model_name)

        self.model = SwinTransformerSys(
            img_size=img_size,
            in_chans=in_channels,
            num_classes=self.num_classes,
            use_checkpoint=False,
            **self.config)

        if pretrained:
            pretrained_path = load_pretrained_model(model_name)
            if pretrained_path:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.load_from(state_dict)
        
    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model

    def forward(self, batch: Dict, device: torch.device):
        x = move_to(batch['inputs'], device)
        outputs = self.model(x)
        return {
            'outputs': outputs,
        }

    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        """
        Inference using the model.
        adict: `Dict[str, Any]`
            dictionary of inputs
        device: `torch.device`
            current device 
        """
        outputs = self.forward(adict, device)['outputs']
        outputs = torch.softmax(outputs, dim=1) # B, C, H, W

        if 'weights' in adict.keys():
            weights = adict['weights'] # C
            for i, weight in enumerate(weights):
                outputs[:, i] *= weight
                
        if self.num_classes == 1:
            thresh = adict['thresh']
            predicts = (outputs > thresh).float()
        else:
            predicts = torch.argmax(outputs, dim=1)

        predicts = move_to(detach(predicts), torch.device('cpu')).squeeze().numpy()
        return {
            'masks': predicts
        } 

    def load_from(self, pretrained_dict):
        pretrained_dict = pretrained_dict['model']
        model_dict = self.model.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3-int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k:v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                    del full_dict[k]

        self.model = load_state_dict(self.model, full_dict, strict=False)