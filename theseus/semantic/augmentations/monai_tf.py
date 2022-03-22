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