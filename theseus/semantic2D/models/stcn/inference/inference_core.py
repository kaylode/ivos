import torch
import numpy as np
from typing import Dict
from theseus.semantic2D.models.stcn.inference.inference_memory_bank import MemoryBank
from theseus.semantic2D.models.stcn.networks.eval_network import STCNEval
from theseus.semantic2D.models.stcn.utilities.aggregate import aggregate
from theseus.semantic2D.models.stcn.utilities.tensor_util import pad_divide_by
from theseus.utilities.cuda import move_to

from theseus.semantic2D.utilities.referencer import Referencer
REFERENCER = Referencer()

class InferenceCore:
    """
    Inference module, which performs iterative propagation
    """
    def __init__(self, prop_net:STCNEval, images, num_objects, top_k=20, mem_every=5, include_last=False,device='cuda'):
        self.prop_net = prop_net
        self.mem_every = mem_every
        self.include_last = include_last

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        images, self.pad = pad_divide_by(images, 16)
        # Padded dimensions
        nh, nw = images.shape[-2:]

        self.images = images
        self.device = device

        self.k = num_objects

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = torch.zeros((self.k, t, 1, nh, nw), dtype=torch.float32)
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16
        self.top_k = top_k

        self.flush_memory(top_k)

    def flush_memory(self, top_k):
        self.mem_bank = MemoryBank(k=self.k-1, top_k=top_k)

    def encode_key(self, idx):
        result = self.prop_net.encode_key(self.images[:,idx].to(self.device))
        return result

    def do_pass(self, key_k, key_v, idx, end_idx):
        self.mem_bank.add_memory(key_k, key_v)

        # Note that we never reach closest_ti, just the frame before it
        if idx < end_idx:
            closest_ti = end_idx
            this_range = range(idx+1, closest_ti)
            end = closest_ti - 1
        else:
            closest_ti = end_idx
            this_range = range(idx, closest_ti-1, -1)
            end = closest_ti + 1

        for ti in this_range:
            k16, qv16, qf16, qf8, qf4 = self.encode_key(ti)
            out_mask = self.prop_net.segment_with_query(self.mem_bank, qf8, qf4, k16, qv16)
            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[:,ti] += move_to(out_mask, torch.device('cpu'))

            if ti != end:
                is_mem_frame = ((ti % self.mem_every) == 0)
                if self.include_last or is_mem_frame:
                    prev_value = self.prop_net.encode_value(self.images[:,ti].to(self.device), qf16, out_mask[1:])
                    prev_key = k16.unsqueeze(2)
                    self.mem_bank.add_memory(prev_key, prev_value, is_temp=not is_mem_frame)

        return closest_ti

    def interact(self, mask, frame_idx, end_idx):
        mask, _ = pad_divide_by(mask.cuda(), 16)
        self.prob[:, frame_idx] += move_to(aggregate(mask, keep_bg=True), torch.device('cpu'))


        # KV pair for the interacting frame
        key_k, _, qf16, _, _ = self.encode_key(frame_idx)
        key_v = self.prop_net.encode_value(self.images[:,frame_idx].to(self.device), qf16, self.prob[1:,frame_idx].to(self.device))
        key_k = key_k.unsqueeze(2)

        # Propagate
        self.do_pass(key_k, key_v, frame_idx, end_idx)

    def get_prediction(self, adict:Dict):

        msk = adict['msk']
        rgb = adict['rgb']
        guide_indices = adict['guide_indices']

        # iter through all reference images and register into memory
        prop_range = REFERENCER.find_propagation_range(guide_indices, length=msk.shape[0])
        for prange in prop_range:
            start_idx, end_idx = prange
            self.interact(msk[:,start_idx], start_idx, end_idx)

        # reverse backprop
        if adict.get('bidirectional', None):
            self.flush_memory(self.top_k) # clear memory
            rev_prop_range = REFERENCER.find_propagation_range(list(reversed(guide_indices)), length=msk.shape[0])
            # iter through all reference images and register into memory
            for prange in rev_prop_range:
                start_idx, end_idx = prange
                self.interact(msk[:,start_idx], start_idx, end_idx)

        # Do unpad -> upsample to original size 
        out_masks = torch.zeros((self.t, 1, *rgb.shape[-2:]), dtype=torch.float32)
        for ti in range(self.t):
            prob = self.prob[:,ti].detach()

            if self.pad[2]+self.pad[3] > 0:
                prob = prob[:,:,self.pad[2]:-self.pad[3],:]
            if self.pad[0]+self.pad[1] > 0:
                prob = prob[:,:,:,self.pad[0]:-self.pad[1]]

            out_masks[ti] = torch.argmax(prob, dim=0)
        
        out_masks = (out_masks.numpy()[:,0]).astype(np.uint8) # (T, H, W)
            
        return {
            'masks': out_masks
        }