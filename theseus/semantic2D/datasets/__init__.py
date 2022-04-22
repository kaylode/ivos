from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from .brats21 import Brats21Dataset, Brats21Testset
DATASET_REGISTRY.register(Brats21Dataset)
DATASET_REGISTRY.register(Brats21Testset)

from .abdomenct1k import (
    AbdomenCT1KTrainDataset,    
    AbdomenCT1KValDataset,
    AbdomenCT1KNormalDataset,
    AbdomenCT1KNormalTestset
)
DATASET_REGISTRY.register(AbdomenCT1KTrainDataset)
DATASET_REGISTRY.register(AbdomenCT1KValDataset)
DATASET_REGISTRY.register(AbdomenCT1KNormalTestset)
DATASET_REGISTRY.register(AbdomenCT1KNormalDataset)