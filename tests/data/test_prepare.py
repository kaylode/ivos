import pytest
import numpy as np
import os 
from tests import DATA_DIR 
from tests.utils import np2nii
import SimpleITK as sitk
import pandas as pd 

def random_tensor(shape, type=np.float32):
    return np.random.rand(*shape).astype(type)

@pytest.mark.order("first")
def test_prepare_data_nii(): 
    for folder in ['TrainImage','TrainMask','Validation','ValImage','ValMask']:
        os.makedirs(os.path.join(DATA_DIR,folder),exist_ok=True)

    train_ids = range(1,3)
    val_ids = range(3,6)
    test_ids = range(1,3)
    train_imPaths = []
    train_lblPaths = []
    val_imPaths = []
    val_lblPaths = []

    for id in train_ids:
        img = random_tensor((256,256,100))
        mask = random_tensor((256,256,100),np.uint8)
        img = np2nii(img)
        mask = np2nii(mask)
        sitk.WriteImage(img, os.path.join(DATA_DIR,'TrainImage','FLARE22_{:04d}_0000.nii.gz'.format(id)))
        sitk.WriteImage(mask, os.path.join(DATA_DIR,'TrainMask','FLARE22_{:04d}.nii.gz'.format(id)))
        train_imPaths.append(os.path.join(DATA_DIR,'TrainImage','FLARE22_{:04d}_0000.nii.gz'.format(id)))
        train_lblPaths.append(os.path.join(DATA_DIR,'TrainMask','FLARE22_{:04d}.nii.gz'.format(id)))
    
    for id in val_ids:
        img = random_tensor((256,256,100))
        mask = random_tensor((256,256,100),np.uint8)
        img = np2nii(img)
        mask = np2nii(mask)
        sitk.WriteImage(img, os.path.join(DATA_DIR,'ValImage','FLARE22_{:04d}_0000.nii.gz'.format(id)))
        sitk.WriteImage(mask, os.path.join(DATA_DIR,'ValMask','FLARE22_{:04d}.nii.gz'.format(id)))
        val_imPaths.append(os.path.join(DATA_DIR,'ValImage','FLARE22_{:04d}_0000.nii.gz'.format(id)))
        val_lblPaths.append(os.path.join(DATA_DIR,'ValMask','FLARE22_{:04d}.nii.gz'.format(id)))

    for id in test_ids:
        img = random_tensor((256,256,100))
        img = np2nii(img)
        sitk.WriteImage(img, os.path.join(DATA_DIR,'Validation','FLARE22_{:04d}_0000.nii.gz'.format(id)))
    train_df = pd.DataFrame({'image':train_imPaths,'label':train_lblPaths})
    val_df = pd.DataFrame({'image':val_imPaths,'label':val_lblPaths})
    train_df.to_csv(os.path.join(DATA_DIR,'train.csv'),index=False)
    val_df.to_csv(os.path.join(DATA_DIR,'val.csv'),index=False)
    