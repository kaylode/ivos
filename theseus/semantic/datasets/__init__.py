from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from .brats21 import Brats21Dataset, Brats21Testset
DATASET_REGISTRY.register(Brats21Dataset)
DATASET_REGISTRY.register(Brats21Testset)

from .abdomenct1k import AbdomenCT1KDataset, AbdomenCT1KTestset
DATASET_REGISTRY.register(AbdomenCT1KDataset)
DATASET_REGISTRY.register(AbdomenCT1KTestset)