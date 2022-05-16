from theseus.semantic2D.trainer import TRAINER_REGISTRY
from .ss_trainer import SemiSupervisedTrainer


TRAINER_REGISTRY.register(SemiSupervisedTrainer)