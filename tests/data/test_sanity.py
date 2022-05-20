import pytest 
from tests import DATA_PRE_DIR
import os.path as osp 
import SimpleITK as sitk
import numpy as np
import pandas as pd 

def type_check(imgPath, img_type):
    img = sitk.ReadImage(imgPath)
    img = sitk.GetArrayFromImage(img)
    if img_type == 'label':
        assert img.dtype == np.uint8
    if img_type == 'image':
        assert img.dtype == np.float32
    del img

@pytest.mark.order("third")
def test_sanity_check_dataset():
    data_folder = DATA_PRE_DIR
    train_csv = pd.read_csv(osp.join(DATA_PRE_DIR,'train.csv'))
    valid_csv = pd.read_csv(osp.join(DATA_PRE_DIR,'val.csv'))

    for i, row in train_csv.iterrows(): 
        imgPath = osp.join(data_folder,row['image'])
        lblPath = osp.join(data_folder,row['label'])
        assert osp.exists(imgPath)
        assert osp.exists(lblPath)
        type_check(imgPath, 'image')
        type_check(lblPath, 'label')
    
    for i, row in valid_csv.iterrows(): 
        imgPath = osp.join(data_folder,row['image'])
        lblPath = osp.join(data_folder,row['label'])
        assert osp.exists(imgPath)
        assert osp.exists(lblPath)
        type_check(imgPath, 'image')
        type_check(lblPath, 'label')
