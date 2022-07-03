from typing import Dict, Any
import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
from theseus.semantic2D.models.transunet.vit_seg_configs import CONFIGS, load_pretrained_model
from theseus.semantic2D.models.transunet.vit_seg_modeling import (
    Transformer, DecoderCup, SegmentationHead, np2th
)

from theseus.utilities.loading import load_state_dict
from theseus.utilities.cuda import move_to, detach

class TransUnet(nn.Module):
    def __init__(
        self, 
        num_classes: int, 
        model_name: str = 'R50-ViT-B_16', 
        img_size: int = 512, 
        in_channels: int = 3,
        pretrained: bool = True,
        vis=False,
        **kwargs):

        super(TransUnet, self).__init__()
        config = CONFIGS[model_name]

        if model_name.find('R50') != -1:
            config.patches.grid = (
                int(img_size / config.patches.size[0]), 
                int(img_size / config.patches.size[1])
            )

        self.num_classes = num_classes
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis, in_channels=in_channels)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.config = config

        if pretrained:
            ckpt_path = load_pretrained_model(model_name)
            state_dict = np.load(ckpt_path)
            self.load_from(weights=state_dict)

    def forward(self, batch: Dict, device: torch.device):
        x = move_to(batch['inputs'], device)
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return {
            'outputs': logits,
        }

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self

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

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                # logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
