from theseus.semantic.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from .flare2022v2 import (
    FLARE22V2STCNTrainDataset, FLARE22V2STCNValDataset,
    FLARE22V2TestDataset, FLARE22V2CoarseMaskTestDataset,
    FLARE22V2LabelledCSVPosDataset, FLARE22V2UnlabelledCSVPosDataset
)

DATASET_REGISTRY.register(FLARE22V2STCNTrainDataset)
DATASET_REGISTRY.register(FLARE22V2STCNValDataset)
DATASET_REGISTRY.register(FLARE22V2TestDataset)
DATASET_REGISTRY.register(FLARE22V2CoarseMaskTestDataset)
DATASET_REGISTRY.register(FLARE22V2LabelledCSVPosDataset)
DATASET_REGISTRY.register(FLARE22V2UnlabelledCSVPosDataset)