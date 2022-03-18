from theseus.base.trainer import TRAINER_REGISTRY 

from .stcn_trainer import STCNTrainer

TRAINER_REGISTRY.register(STCNTrainer)