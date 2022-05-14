from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from .brats21 import Brats21Dataset, Brats21Testset
DATASET_REGISTRY.register(Brats21Dataset)
DATASET_REGISTRY.register(Brats21Testset)

from .abdomenct1k import (
    AbdomenCT1KTrainDataset,    
    AbdomenCT1KValDataset,
    AbdomenCT1KNormalDataset,
    AbdomenCT1KNormalValDataset,
    AbdomenCT1KTestDataset,
    ReSampler
)
DATASET_REGISTRY.register(AbdomenCT1KTrainDataset)
DATASET_REGISTRY.register(AbdomenCT1KValDataset)
DATASET_REGISTRY.register(AbdomenCT1KNormalValDataset)
DATASET_REGISTRY.register(AbdomenCT1KNormalDataset)
DATASET_REGISTRY.register(AbdomenCT1KTestDataset)

DATALOADER_REGISTRY.register(ReSampler)

from .flare2022 import (
    FLARE22TrainDataset,    
    FLARE22ValDataset,
    FLARE22NormalDataset,
    FLARE22NormalValDataset,
    FLARE22TestDataset,
)
DATASET_REGISTRY.register(FLARE22TrainDataset)
DATASET_REGISTRY.register(FLARE22ValDataset)
DATASET_REGISTRY.register(FLARE22NormalValDataset)
DATASET_REGISTRY.register(FLARE22NormalDataset)
DATASET_REGISTRY.register(FLARE22TestDataset)