%cd main
import SimpleITK as sitk
import shutil
import os
import os.path as osp
import numpy as np
import argparse

parser = argparse.ArgumentParser("Process volume CT")
parser.add_argument("-i", "--input_dir", type=str, help="Volume directory")
parser.add_argument("-o", "--out_dir", type=str, help="Output directory")

def copy_BraTS_segmentation_and_convert_labels(in_file, out_file):
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)

def processing(root_dir, out_dir):

    target_imagesTr = osp.join(out_dir, "imagesTr")
    target_labelsTr = osp.join(out_dir, "labelsTr")
    os.makedirs(target_imagesTr, exist_ok=True)
    os.makedirs(target_labelsTr, exist_ok=True)

    patient_names = []
    folder_names = os.listdir(root_dir)
    for folder_name in folder_names:
        patdir = osp.join(root_dir, folder_name)
        patient_name = folder_name.split('_')[1]
        patient_names.append(patient_name)
        t1 = osp.join(patdir, folder_name + "_t1.nii.gz")
        t1c = osp.join(patdir, folder_name + "_t1ce.nii.gz")
        t2 = osp.join(patdir, folder_name + "_t2.nii.gz")
        flair = osp.join(patdir, folder_name + "_flair.nii.gz")
        seg = osp.join(patdir, folder_name + "_seg.nii.gz")

        os.makedirs(osp.join(target_imagesTr, patient_name), exist_ok=True)
        shutil.copy(t1, osp.join(target_imagesTr, patient_name, patient_name + "_t1.nii.gz"))
        shutil.copy(t1c, osp.join(target_imagesTr, patient_name, patient_name + "_t1ce.nii.gz"))
        shutil.copy(t2, osp.join(target_imagesTr, patient_name, patient_name + "_t2.nii.gz"))
        shutil.copy(flair, osp.join(target_imagesTr, patient_name, patient_name + "_flair.nii.gz"))

        copy_BraTS_segmentation_and_convert_labels(seg, osp.join(target_labelsTr, patient_name + "_seg.nii.gz"))


if __name__ == '__main__':
    args = parser.parse_args()
    processing(args.input_dir, args.out_dir)