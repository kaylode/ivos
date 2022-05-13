import os.path as osp
import nibabel as nib # common way of importing nibabel
import torch
import numpy as np
import pandas as pd
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger('main')

def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for k, l in enumerate(labels):
        Ms[k] = (masks == l).astype(np.uint8)
        
    return Ms

class AbdomenCT1KBaseDataset(torch.utils.data.Dataset):
    """
    Dataset to load AbdomenCT1k data.

    abdomenct1k
    ├── TrainImage
    │   └── <file1>.nii.gz 
    │   ├── <file2>
    ....
    ├── TrainMask
    │   └── <file1>.nii.gz

    root_dir: `str`
        path to root_dir
    """
    def __init__(
        self, 
        root_dir: str,
        transform=None,
        **kwargs):

        self.root_dir = root_dir
        self.transform = transform
        self.classnames = [
            "background",
            "liver", 
            "kidney",
            "spleen",
            "pancreas"
        ]

        self.num_classes = len(self.classnames)

    def _load_item(self, patient_item):
        """
        Load volume with Monai transform
        """
        vol_path = osp.join(self.root_dir, patient_item['vol'])

        if 'label' in patient_item.keys():
            gt_path = osp.join(self.root_dir, patient_item['label'])
            nib_label = nib.load(gt_path)
            affine = nib_label.affine
            case_spacing = nib_label.header.get_zooms()

            out_dict = self.transform({
                'image': [vol_path],
                'label': [gt_path],
                'affine': affine, 
                'spacing':  case_spacing
            })

        else:
            nib_label = nib.load(vol_path)
            affine = nib_label.affine
            case_spacing = nib_label.header.get_zooms()
            out_dict = self.transform({
                'image': [vol_path],
                'affine': affine, 
                'spacing':  case_spacing
            })

        return out_dict

    def __len__(self):
        return len(self.fns)

class AbdomenCT1KBaseCSVDataset(AbdomenCT1KBaseDataset):
    """
    Load in csv
    """
    def __init__(
        self, 
        root_dir: str,
        csv_path: str,
        transform=None,
        **kwargs):

        super().__init__(root_dir, transform)
        self.csv_path = csv_path
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.csv_path)
        self.fns = []
        for _, row in df.iterrows():
            # train_0047_0000.nii.gz
            volume_name, mask_name = row
            pid = volume_name.split('_')[1] 
            self.fns.append({
                'pid': pid,
                'vol': volume_name,
                'label': mask_name,
            })