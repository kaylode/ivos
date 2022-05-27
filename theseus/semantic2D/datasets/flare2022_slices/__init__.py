from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from .base import FLARE22SlicesBaseCSVDataset, all_to_onehot
from .flare22 import FLARE22SlicesDataset, FLARE22SlicesNormalDataset

DATASET_REGISTRY.register(FLARE22SlicesDataset)
DATASET_REGISTRY.register(FLARE22SlicesNormalDataset)
