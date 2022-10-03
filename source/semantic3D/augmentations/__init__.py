from theseus.semantic.augmentations import TRANSFORM_REGISTRY

from monai.transforms import (
    Compose,
    LoadImaged,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd, 
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
    CenterSpatialCropd,
)
from .monai_tf import *

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

TRANSFORM_REGISTRY.register(LoadImageAndResize3D, prefix='Monai')
TRANSFORM_REGISTRY.register(PercentileClip, prefix='Monai')