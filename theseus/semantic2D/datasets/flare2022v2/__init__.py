from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from .testset import FLARE22V2TestDataset, FLARE22V2CoarseMaskTestDataset
from .flare22v2_normal import FLARE22V2LabelledCSVDataset, FLARE22V2UnlabelledCSVDataset
from .flare22v2_stcn import FLARE22V2STCNTrainDataset, FLARE22V2STCNValDataset