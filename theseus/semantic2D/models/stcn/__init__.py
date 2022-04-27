from typing import Dict, Any
import torch
import torch.nn as nn
import numpy as np
from theseus.utilities.cuda import move_to
from theseus.utilities.loading import load_state_dict
from theseus.semantic2D.models.stcn.networks.network import STCNTrain
from theseus.semantic2D.models.stcn.inference.inference_core import InferenceCore
from theseus.semantic2D.models.stcn.networks.eval_network import STCNEval
from theseus.semantic2D.models.stcn.utilities.loading import load_pretrained_model


class STCNModel():
    """
    STCN Wrapper
    """
    def __init__(
        self, 
        num_classes: int = 3,
        classnames: str = None,
        single_object: bool = False,
        top_k_eval: int = 20,
        mem_every_eval: int = 5,
        pretrained: bool = False,
        **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.classnames = classnames
        self.single_object = single_object
        self.top_k_eval = top_k_eval
        self.mem_every_eval = mem_every_eval

        self.train_model = STCNTrain(self.single_object)
        self.eval_model = STCNEval()
        self.training = False
        self.pretrained = pretrained

        if self.pretrained:
            pretrained_path = load_pretrained_model('stcn')
            if pretrained_path:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.load_network(state_dict)
        
    def state_dict(self):
        return self.train_model.state_dict()

    def load_state_dict(self, state_dict):
        self.train_model = load_state_dict(self.train_model , state_dict)

    def get_model(self):
        return self.train_model

    def eval(self):
        self.train_model.to(torch.device('cpu'))
        self.eval_model.to(torch.device('cuda'))
        self.eval_model.load_state_dict(self.train_model.state_dict())
        self.training = False

    def train(self):
        self.train_model.to(torch.device('cuda'))
        self.eval_model.to(torch.device('cpu'))
        self.training = True

    def __call__(self, data:Dict):
        if self.training:
            return self.forward_train(data)
        else:
            return self.forward_val(data)

    @torch.no_grad()
    def forward_val(self, data: Dict):
        torch.set_grad_enabled(False)

        rgb = data['inputs'].float().cuda()
        msk = data['gt'][0].cuda()
        info = data['info']
        guidemark = info['guidemark']
        k = len(info['labels'][0])

        self.processor = InferenceCore(
            self.eval_model, rgb, k, 
            top_k=self.top_k_eval, 
            mem_every=self.mem_every_eval
        )

        out_masks = self.processor.get_prediction({
            'rgb': rgb,
            'msk': msk,
            'prop_range': [(0, int(guidemark)), (int(guidemark), rgb.shape[1])] # reference guide frame index, 0 because we already process in the dataset
        })['masks']

        first = out_masks[:guidemark, :, :]
        second = out_masks[guidemark:, :, :]
        second = np.flip(second, axis=0)

        out_masks = np.concatenate([second, first[1:,:,:]], axis=0)

        del rgb
        del msk
        del self.processor

        return {
            'outputs': out_masks
        }

    def forward_train(self, data: Dict):
        # No need to store the gradient outside training
        torch.set_grad_enabled(True)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        Fs = data['inputs'].float()
        Ms = data['targets']
        
        # key features never change, compute once
        k16, kf16_thin, kf16, kf8, kf4 = self.train_model('encode_key', Fs)

        if self.single_object:
            ref_v = self.train_model('encode_value', Fs[:,0], kf16[:,0], Ms[:,0])

            # Segment frame 1 with frame 0
            prev_logits, prev_mask = self.train_model('segment', 
                    k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                    k16[:,:,0:1], ref_v)
            prev_v = self.train_model('encode_value', Fs[:,1], kf16[:,1], prev_mask)

            values = torch.cat([ref_v, prev_v], 2)

            del ref_v

            # Segment frame 2 with frame 0 and 1
            this_logits, this_mask = self.train_model('segment', 
                    k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                    k16[:,:,0:2], values)

            out['mask_1'] = prev_mask
            out['mask_2'] = this_mask
            out['logits_1'] = prev_logits
            out['logits_2'] = this_logits
        else:
            sec_Ms = data['sec_gt']
            selector = data['selector']

            ref_v1 = self.train_model('encode_value', Fs[:,0], kf16[:,0], Ms[:,0], sec_Ms[:,0])
            ref_v2 = self.train_model('encode_value', Fs[:,0], kf16[:,0], sec_Ms[:,0], Ms[:,0])
            ref_v = torch.stack([ref_v1, ref_v2], 1)

            # Segment frame 1 with frame 0
            prev_logits, prev_mask = self.train_model('segment', 
                    k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                    k16[:,:,0:1], ref_v, selector)
            
            prev_v1 = self.train_model('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,0:1], prev_mask[:,1:2])
            prev_v2 = self.train_model('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,1:2], prev_mask[:,0:1])
            prev_v = torch.stack([prev_v1, prev_v2], 1)
            values = torch.cat([ref_v, prev_v], 3)

            del ref_v

            # Segment frame 2 with frame 0 and 1
            this_logits, this_mask = self.train_model('segment', 
                    k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                    k16[:,:,0:2], values, selector)

            out['mask_1'] = prev_mask[:,0:1]
            out['mask_2'] = this_mask[:,0:1]
            out['sec_mask_1'] = prev_mask[:,1:2]
            out['sec_mask_2'] = this_mask[:,1:2]

            out['logits_1'] = prev_logits
            out['logits_2'] = this_logits

        return out

    def load_network(self, state_dict):
        # This method loads only the network weight and should be used to load a pretrained model

        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(state_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if state_dict[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=state_dict[k].device)
                    nn.init.orthogonal_(pads)
                    state_dict[k] = torch.cat([state_dict[k], pads], 1)

        self.train_model = load_state_dict(self.train_model, state_dict)