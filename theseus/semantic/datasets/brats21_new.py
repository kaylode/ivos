import os
import os.path as osp
import nibabel as nib # common way of importing nibabel

import torch
import numpy as np

from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')

def normalize_min_max(data):
    data = (data - np.min(data)) / np.max(data)
    return data

def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for k, l in enumerate(labels):
        Ms[k] = (masks == l).astype(np.uint8)
        
    return Ms

class Brats21Dataset(torch.utils.data.Dataset):
    """
    Generate pseudo VOS data by applying random transforms on static images.
    Single-object only.

    brats21
    ├── imagesTr
    │   ├── <patient1's id>
    │   │   └── <patient1's id>_t1.nii.gz 
    │   │   └── <patient1's id>_t1ce.nii.gz 
    │   │   └── <patient1's id>_t2.nii.gz 
    │   │   └── <patient1's id>_flair.nii.gz 
    │   ├── <patient2's id>
    ....
    ├── labelsTr
    │   └── <patient's id>.nii.gz


    """
    def __init__(self, volume_dir, label_dir, target_shape=240, max_jump=25, transform=None):
        self.volume_dir = volume_dir
        self.label_dir = label_dir
        self.target_shape = target_shape
        self.max_jump = max_jump
        self.transform = transform

        self.patient_ids = os.listdir(self.volume_dir)
        self.fns = []
        for pid in self.patient_ids:
            self.fns.append({
                'pid': pid,
                't1': pid+"_t1.nii.gz",
                't1ce': pid+"_t1ce.nii.gz",
                't2': pid+"_t2.nii.gz",
                'flair': pid+"_flair.nii.gz",
                'label': pid+"_seg.nii.gz",
            })


        self.channel_names = ['t1', 't1ce', 't2'] #'flair' 
        self.num_channels = len(self.channel_names)
        self.classnames = [
            "background",
            "edema",
            "non-enhancing",
            "enhancing"
        ]

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
            gt_path = osp.join(self.label_dir, item['label'])
            gt_vol = nib.load(gt_path).get_fdata()# (H, W, NS)
            num_slices = gt_vol.shape[-1]
            
            # Search for guide frames, in which all classes are presented
            max_possible_number_of_classes = len(np.unique(gt_vol))
            for frame_idx in range(num_slices):
                num_classes = len(np.unique(gt_vol[:, :, frame_idx]))
                if num_classes == max_possible_number_of_classes:
                    vol_dict['guides'].append(frame_idx)
            self.stats.append(vol_dict)

    def load_item(self, patient_item):
        """
        Load volume with Monai transform
        """
        patient_id = patient_item['pid']
        vol_paths = []
        for c in self.channel_names:
            vol_path = osp.join(self.volume_dir, patient_id, patient_item[c])
            vol_paths.append(vol_path)

        gt_path = osp.join(self.label_dir, patient_item['label'])

        out_dict = self.transform({
            'image': vol_paths,
            'label': [gt_path]
        })

        return out_dict['image'], out_dict['label']

    def __getitem__(self, idx):
        
        patient_item = self.fns[idx]
        patient_id = patient_item['pid']
        stacked_vol, gt_vol = self.load_item(patient_item)

        num_slices = stacked_vol.shape[-1]

        # Item stats
        # stat = self.stats[idx]

        trials = 0
        while trials < 5:

            # Don't want to bias towards beginning/end
            this_max_jump = min(num_slices, self.max_jump)
            # guide_idx = np.random.randint(len(stat['guides']))
            # start_idx = stat['guides'][guide_idx]
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
                this_im = stacked_vol[:,:,:,f_idx] #(4, H, W)
                this_gt = gt_vol[:,:,f_idx] #(H, W)
                this_im = torch.from_numpy(this_im)
                this_gt = np.array(this_gt)

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

        cls_gt = np.zeros((self.num_channels, self.target_shape, self.target_shape), dtype=np.int)
        cls_gt[tar_masks[:,0] > 0.5] = 1
        cls_gt[sec_masks[:,0] > 0.5] = 2

        data = {
            'inputs': images, # normalized image, torch.Tensor (T, C, H, W) 
            'targets': tar_masks, # target mask, torch.Tensor (T, 1, H, W) , values 1 at primary class
            'cls_gt': cls_gt, # numpy (T, H, W), each pixel present one class
            'sec_gt': sec_masks, # second object mask, torch.Tensor (T, 1, H, W) , values 1 at second class
            'selector': selector, # [1, 1] if has second object, else [1, 0]
            'info': {
                'name': patient_id,
                'slice_id': frames_idx
            },
        }

        return data

    def __len__(self):
        return len(self.fns)

class Brats21Testset(Brats21Dataset):
    def __init__(self, volume_dir, label_dir, target_shape=240):
        super().__init__(volume_dir, label_dir, target_shape, 0)
        self.single_object = False


    def __getitem__(self, idx):
        patient_item = self.fns[idx]
        patient_id = patient_item['pid']
        stacked_channels = []
        for c in self.channel_names:
            vol_path = osp.join(self.volume_dir, patient_id, patient_item[c])
            img = nib.load(vol_path)
            img_data = img.get_fdata()
            stacked_channels.append(img_data)

        stacked_vol = np.stack(stacked_channels, axis=0) # (4, H, W, NS)
        stacked_vol = normalize_min_max(stacked_vol)
        images = torch.from_numpy(stacked_vol).permute(3, 0, 1, 2) # (C, H, W, NS) --> (NS, C, H, W)
        
        stat = self.stats[idx]
        guide_id = np.random.choice(stat['guides'])
        first = images[guide_id:, :, :, :]

        guidemark = first.shape[0]
        second = images[:guide_id+1, :, :, :]
        second = torch.flip(second, dims=[0])
        images = torch.cat([first, second], dim=0)        

        num_slices = images.shape[0]

        gt_path = osp.join(self.label_dir, patient_item['label'])
        masks = []
        
        nib_label = nib.load(gt_path)
        affine = nib_label.affine
        gt_vol = nib_label.get_fdata()# (H, W, NS)
        gt_vol1 = gt_vol[:, :, guide_id:]
        gt_vol2 = gt_vol[:, :, :guide_id+1]

        gt_vol2 = np.flip(gt_vol2, axis=-1)
        gt_vol = np.concatenate([gt_vol1, gt_vol2], axis=-1)        

        for f in range(num_slices):
            # Test-set maybe?
            if f==0 or f == guidemark:
                masks.append(gt_vol[:,:,f])
            else:
                masks.append(np.zeros_like(masks[0]))
        
        masks = np.stack(masks, 0)

        if self.single_object:
            labels = [1]
            masks = (masks > 0.5).astype(np.uint8)
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()
        else:
            labels = np.unique(masks)
            labels = labels[labels!=0]
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        masks = masks.unsqueeze(2)

        data = {
            'inputs': images,
            'targets': masks,
            'info': {
                'name': vol_path,
                'labels': labels,
                'guide_id': guide_id,
                'guidemark': guidemark,
                'affine': affine
            },
        }

        return data