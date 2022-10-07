from .cps_loss import CPSLoss
from .uncertainty import UncertaintyCPSLoss
from source.semantic2D.losses import LOSS_REGISTRY

LOSS_REGISTRY.register(CPSLoss)
LOSS_REGISTRY.register(UncertaintyCPSLoss)