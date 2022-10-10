from theseus.semantic.losses import LOSS_REGISTRY

from .stcn_loss import STCNLoss, STCNLossV2
from .abl import ABL

LOSS_REGISTRY.register(STCNLoss)
LOSS_REGISTRY.register(STCNLossV2)
LOSS_REGISTRY.register(ABL)
