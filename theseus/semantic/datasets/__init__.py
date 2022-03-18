from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from .brats21 import Brats21Dataset
DATASET_REGISTRY.register(Brats21Dataset)

from .stcn_loader import DistributedDataloader
DATALOADER_REGISTRY.register(DistributedDataloader)