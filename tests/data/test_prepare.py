import pytest
import numpy as np
import os 
from tests import DATA_RAW_DIR 
from tests.utils import np2nii
import SimpleITK as sitk
import pandas as pd 
from tests.utils import run_cmd

def random_img(shape):
    return np.random.rand(*shape).astype(np.float32)

def random_msk(shape, N):
    return np.random.randint(0, N, shape).astype(np.uint8)

@pytest.mark.order("first")
def test_prepare_data_nii(): 
    for folder in ['TrainImage','TrainMask','Validation']:
        os.makedirs(os.path.join(DATA_RAW_DIR,folder),exist_ok=True)

    train_ids = range(1,3)
    test_ids = range(1,2)
    NUM_CLS = 14
    train_imPaths = []
    train_lblPaths = []

    for id in train_ids:
        img = random_img((256,256,100))
        mask = random_msk((256,256,100),N=NUM_CLS)
        img = np2nii(img)
        mask = np2nii(mask)
        sitk.WriteImage(img, os.path.join(DATA_RAW_DIR,'TrainImage','FLARE22_{:04d}_0000.nii.gz'.format(id)))
        sitk.WriteImage(mask, os.path.join(DATA_RAW_DIR,'TrainMask','FLARE22_{:04d}.nii.gz'.format(id)))
        train_imPaths.append(os.path.join(DATA_RAW_DIR,'TrainImage','FLARE22_{:04d}_0000.nii.gz'.format(id)))
        train_lblPaths.append(os.path.join(DATA_RAW_DIR,'TrainMask','FLARE22_{:04d}.nii.gz'.format(id)))

    for id in test_ids:
        img = random_img((256,256,100))
        img = np2nii(img)
        sitk.WriteImage(img, os.path.join(DATA_RAW_DIR,'Validation','FLARE22_{:04d}_0000.nii.gz'.format(id)))

@pytest.mark.order("second")
def test_preproc():
    run_cmd("python tools/preprocess/make_flare22.py \
            -i data/sample/ \
            -o data/sample_binary/ \
            --ratio 0.5 --binary", "test_preproc")