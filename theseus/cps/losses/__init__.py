from .cps_loss import CPSLoss
from theseus.semantic2D.losses import LOSS_REGISTRY

LOSS_REGISTRY.register(CPSLoss)