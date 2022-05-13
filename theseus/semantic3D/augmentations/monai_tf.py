from typing import List
import SimpleITK as sitk

from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    RandSpatialCropd,
    RandFlipd,
    MapTransform,
    NormalizeIntensityd, 
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
    CenterSpatialCropd,
)
import numpy as np


class UnsqueezeOnAxisd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    """

    def __init__(self, keys, allow_missing_keys: bool = False, axis: int =0) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.axis = axis
 
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.expand_dims(d[key], axis=self.axis)
        return d

from theseus.base.augmentations import TRANSFORM_REGISTRY

TRANSFORM_REGISTRY.register(Compose, prefix='Monai')
TRANSFORM_REGISTRY.register(LoadImaged, prefix='Monai')
TRANSFORM_REGISTRY.register(RandSpatialCropd, prefix='Monai')
TRANSFORM_REGISTRY.register(CenterSpatialCropd, prefix='Monai')
TRANSFORM_REGISTRY.register(RandFlipd, prefix='Monai')
TRANSFORM_REGISTRY.register(NormalizeIntensityd, prefix='Monai')
TRANSFORM_REGISTRY.register(RandScaleIntensityd, prefix='Monai')
TRANSFORM_REGISTRY.register(RandShiftIntensityd, prefix='Monai')
TRANSFORM_REGISTRY.register(ToTensord, prefix='Monai')
TRANSFORM_REGISTRY.register(UnsqueezeOnAxisd, prefix='Monai')

class LoadImageAndResize3D(MapTransform):
    """
    Resize 3D CT Volume
    Arguments:

    :size:
        (H,W,C)
    """
    def __init__(self, keys, size: List[int], allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.size = size

    def _resize_image(sefl, itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
        resampler = sitk.ResampleImageFilter()
        originSize = itkimage.GetSize()  
        originSpacing = itkimage.GetSpacing()
        newSize = np.array(newSize,float)
        factor = originSize / newSize
        newSpacing = originSpacing * factor
        newSize = newSize.astype(int) 
        resampler.SetReferenceImage(itkimage) 
        resampler.SetSize(newSize.tolist())
        resampler.SetOutputSpacing(newSpacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itkimgResampled = resampler.Execute(itkimage)  

        return itkimgResampled

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = sitk.ReadImage(d[key])
            if key == 'label':
                resize_img=self._resize_image(image, self.size,resamplemethod= sitk.sitkNearestNeighbor)
            else:
                resize_img=self._resize_image(image, self.size,resamplemethod= sitk.sitkLinear)
            resize_im=sitk.GetArrayFromImage(resize_img)
            d[key] = resize_im

        return d

class PercentileClip(MapTransform):
    """
    Perform clipping on volume CT by intensity percentile
    """
    def __init__(self, keys, min_pct: int, max_pct: int, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.min_pct = min_pct
        self.max_pct = max_pct

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            top = np.percentile(d[key], self.max_pct)
            bottom = np.percentile(d[key], self.min_pct)
            d[key] = np.clip(d[key],bottom, top)

        return d

class IntensityClip(MapTransform):
    """
    Perform clipping on volume CT by intensity value
    """
    def __init__(self, keys, min_value: int, max_value: int, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            d[key] = np.clip(d[key],self.min_value, self.max_value)

        return d

TRANSFORM_REGISTRY.register(LoadImageAndResize3D, prefix='Monai')
TRANSFORM_REGISTRY.register(PercentileClip, prefix='Monai')