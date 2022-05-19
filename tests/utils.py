# import sitk 

import SimpleITK as sitk
def np2nii(np_array):
    result_image = sitk.GetImageFromArray(np_array)
    return result_image
