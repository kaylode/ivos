import os.path as osp
import nibabel as nib # common way of importing nibabel
import torch
import numpy as np
from theseus.semantic2D.datasets.abdomenct1k import AbdomenCT1KBaseCSVDataset
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')


class AbdomenCT1KTrainDataset(AbdomenCT1KBaseCSVDataset):
    """
    Generate pseudo VOS data by applying random transforms on static images.
    Single-object only.

    abdomenct1k
    ├── TrainImage
    │   └── <file1>.nii.gz 
    │   ├── <file2>
    ....
    ├── TrainMask
    │   └── <file1>.nii.gz

    volume_dir: `str`
        path to `TrainImage`
    label_dir: `str`
        path to `TrainMask`
    max_jump: `int`
        max number of frames to jump
    """
    def __init__(
        self, 
        root_dir: str,
        csv_path: str,
        max_jump: int=25,
        transform=None,
        **kwargs):

        super().__init__(root_dir, csv_path, transform)
        self.max_jump = max_jump

    def __getitem__(self, idx):
        
        patient_item = self.fns[idx]
        patient_id = patient_item['pid']
        item_dict = self._load_item(patient_item) # torch.Size([C, H, W, T]), torch.Size([1, H, W, T])
        
        image = item_dict['image']
        gt_vol = item_dict['label']
        affine = item_dict['affine']
        case_spacing = item_dict['spacing']
        stacked_vol = torch.stack([image, image, image], dim=0)

        gt_vol = gt_vol.squeeze(0)
        _, h, w, num_slices = stacked_vol.shape

        trials = 0
        while trials < 5:

            # Don't want to bias towards beginning/end
            this_max_jump = min(num_slices, self.max_jump)
            start_idx = np.random.randint(num_slices-this_max_jump+1)
            f1_idx = start_idx + np.random.randint(this_max_jump+1) + 1
            f1_idx = min(f1_idx, num_slices-this_max_jump, num_slices-1)

            f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
            f2_idx = min(f2_idx, num_slices-this_max_jump//2, num_slices-1)

            frames_idx = [start_idx, f1_idx, f2_idx]
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            images = []
            masks = []
            target_object = None
            for f_idx in frames_idx:
                this_im = stacked_vol[:,:,:,f_idx] #(C, H, W)
                this_gt = gt_vol[:,:,f_idx] #(H, W)
                this_gt = this_gt.numpy()

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]
            
            if len(labels) == 0:
                target_object = -1 # all black if no objects
                has_second_object = False
                trials += 1
            else:
                target_object = np.random.choice(labels)
                has_second_object = (len(labels) > 1)
                if has_second_object:
                    labels = labels[labels!=target_object]
                    second_object = np.random.choice(labels)
                break

        masks = np.stack(masks, 0)
        tar_masks = (masks==target_object).astype(np.float32)[:,np.newaxis,:,:]
        if has_second_object:
            sec_masks = (masks==second_object).astype(np.float32)[:,np.newaxis,:,:]
            selector = torch.FloatTensor([1, 1])
        else:
            sec_masks = np.zeros_like(tar_masks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((3, h, w), dtype=np.int)
        cls_gt[tar_masks[:,0] > 0.5] = 1
        cls_gt[sec_masks[:,0] > 0.5] = 2

        data = {
            'inputs': images, # normalized image, torch.Tensor (T, C, H, W) 
            'targets': tar_masks, # target mask, numpy (T, 1, H, W) , values 1 at primary class
            'cls_gt': cls_gt, # numpy (T, H, W), each pixel present one class
            'sec_gt': sec_masks, # second object mask, numpy (T, 1, H, W) , values 1 at second class
            'selector': selector, # [1, 1] if has second object, else [1, 0]
            'info': {
                'name': patient_id,
                'slice_id': frames_idx,
                'affine': affine,
                'case_spacing': case_spacing
            },
        }

        return data

    def __len__(self):
        return len(self.fns)

class AbdomenCT1KValDataset(AbdomenCT1KBaseCSVDataset):
    def __init__(
        self, 
        root_dir: str,
        csv_path: str,
        transform=None,
        **kwargs):

        super().__init__(root_dir, csv_path, transform)
        self.single_object = False
        self.compute_stats()

    def compute_stats(self):
        """
        Compute statistic for dataset
        """
        
        LOGGER.text("Computing statistic...", level=LoggerObserver.INFO)

        self.stats = []
        for item in self.fns:

            vol_dict = {
                'guides': [],
                'num_labels': []
            }

            patient_id = item['pid']
            gt_path = osp.join(self.root_dir, item['label'])
            gt_vol = nib.load(gt_path).get_fdata()# (H, W, NS)
            num_slices = gt_vol.shape[-1]
            
            # Search for guide frames, in which most classes are presented
            max_possible_number_of_classes = 0
            for frame_idx in range(num_slices):
                num_classes = len(np.unique(gt_vol[:, :, frame_idx]))
                if num_classes == max_possible_number_of_classes:
                    vol_dict['guides'].append(frame_idx)
                elif num_classes > max_possible_number_of_classes:
                    max_possible_number_of_classes = num_classes
                    vol_dict['guides'] = [frame_idx]

            self.stats.append(vol_dict)

    def __getitem__(self, idx):
        patient_item = self.fns[idx]
        patient_id = patient_item['pid']
        item_dict = self._load_item(patient_item) # torch.Size([C, H, W, T]), torch.Size([1, H, W, T])
        image = item_dict['image']
        ori_vol = item_dict['label']
        affine = item_dict['affine']
        case_spacing = item_dict['spacing']
        stacked_vol = torch.stack([image, image, image], dim=0)

        images = stacked_vol.permute(3, 0, 1, 2) # (C, H, W, NS) --> (NS, C, H, W)
        
        # Choose a reference frame
        stat = self.stats[idx]
        guide_id = np.random.choice(stat['guides'])
        first = images[guide_id:, :, :, :]

        # Split in half, flip first half
        guidemark = first.shape[0]
        second = images[:guide_id+1, :, :, :]
        second = torch.flip(second, dims=[0])
        images = torch.cat([first, second], dim=0)        

        data = {
            'inputs': images, # (num_slices+1, C, H, W)
            'targets': ori_vol.squeeze().numpy(), # for evaluation (1, H, W, num_slices)
            'info': {  # infos are used for validation and inference
                'name': patient_id,
                'guide_id': guide_id,       # guide id frame used for reconvert
                'guidemark': guidemark,     # 
                'affine': affine, # from nib.load
                'case_spacing': case_spacing
            },
        }

        return data