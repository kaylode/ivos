import torch
import numpy as np
from typing import Dict
from theseus.semantic2D.models.stcn.inference.inference_memory_bank import MemoryBank
from theseus.semantic2D.models.stcn.networks.eval_network import STCNEval
from theseus.semantic2D.models.stcn.utilities.aggregate import aggregate
from theseus.semantic2D.models.stcn.utilities.tensor_util import pad_divide_by


class InferenceCore:
    def __init__(self, prop_net:STCNEval, images, num_objects, top_k=20, mem_every=5, include_last=False):
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
        self.device = 'cuda'

        self.k = num_objects

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = torch.zeros((self.k, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        self.mem_bank = MemoryBank(k=self.k-1, top_k=top_k)

    def encode_key(self, idx):
        result = self.prop_net.encode_key(self.images[:,idx].cuda())
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
            self.prob[:,ti] = out_mask

            # In-place fusion, maximizes the use of queried buffer
            # esp. for long sequence where the buffer will be flushed
            # if (closest_ti != self.t) and (closest_ti != -1):
            #     self.prob[:,ti] = self.fuse_one_frame(closest_ti, idx, ti, self.prob[:,ti], out_mask, 
            #                             key_k, query[3]).to(self.result_dev)
            # else:
            #     self.prob[:,ti] = out_mask.to(self.result_dev)

            if ti != end:
                is_mem_frame = ((ti % self.mem_every) == 0)
                if self.include_last or is_mem_frame:
                    prev_value = self.prop_net.encode_value(self.images[:,ti].cuda(), qf16, out_mask[1:])
                    prev_key = k16.unsqueeze(2)
                    self.mem_bank.add_memory(prev_key, prev_value, is_temp=not is_mem_frame)

        return closest_ti

    # def fuse_one_frame(self, tc, tr, ti, prev_mask, curr_mask, mk16, qk16):
    #     assert(tc<ti<tr or tr<ti<tc)

    #     prob = torch.zeros((self.k, 1, self.nh, self.nw), dtype=torch.float32, device=self.device)

    #     # Compute linear coefficients
    #     nc = abs(tc-ti) / abs(tc-tr)
    #     nr = abs(tr-ti) / abs(tc-tr)
    #     dist = torch.FloatTensor([nc, nr]).to(self.device).unsqueeze(0)
    #     for k in range(1, self.k+1):
    #         attn_map = self.prop_net.get_attention(mk16[k-1:k], self.pos_mask_diff[k:k+1], self.neg_mask_diff[k:k+1], qk16)

    #         w = torch.sigmoid(self.fuse_net(self.get_image_buffered(ti), 
    #                 prev_mask[k:k+1].to(self.device), curr_mask[k:k+1].to(self.device), attn_map, dist))
    #         prob[k-1] = w 
    #     return aggregate_wbg(prob, keep_bg=True)

    def interact(self, mask, frame_idx, end_idx):
        mask, _ = pad_divide_by(mask.cuda(), 16)
        self.prob[:, frame_idx] = aggregate(mask, keep_bg=True)

        # KV pair for the interacting frame
        key_k, _, qf16, _, _ = self.encode_key(frame_idx)
        key_v = self.prop_net.encode_value(self.images[:,frame_idx].cuda(), qf16, self.prob[1:,frame_idx].cuda())
        key_k = key_k.unsqueeze(2)

        # Propagate
        self.do_pass(key_k, key_v, frame_idx, end_idx)

    def get_prediction(self, adict:Dict):

        msk = adict['msk']
        rgb = adict['rgb']
        prop_range = adict['prop_range']

        # iter through all reference images and register into memory
        for prange in prop_range:
            start_idx, end_idx = prange
            self.interact(msk[:,start_idx], start_idx, end_idx)

        # Do unpad -> upsample to original size 
        out_masks = torch.zeros((self.t, 1, *rgb.shape[-2:]), dtype=torch.float32, device=self.device)
        for ti in range(self.t):
            prob = self.prob[:,ti]

            if self.pad[2]+self.pad[3] > 0:
                prob = prob[:,:,self.pad[2]:-self.pad[3],:]
            if self.pad[0]+self.pad[1] > 0:
                prob = prob[:,:,:,self.pad[0]:-self.pad[1]]

            out_masks[ti] = torch.argmax(prob, dim=0)
        
        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8) # (T, H, W)
            
        return {
            'masks': out_masks
        }