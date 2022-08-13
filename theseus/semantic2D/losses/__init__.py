from theseus.base.losses import LOSS_REGISTRY

from .ce_loss import *
from .tversky_loss import FocalTverskyLoss
from .dice_loss import DiceLoss
from .stcn_loss import STCNLoss, STCNLossV2
from .abl import ABL
from .lovasz_loss import LovaszSoftmax

LOSS_REGISTRY.register(CELoss)
LOSS_REGISTRY.register(LovaszSoftmax)
LOSS_REGISTRY.register(OhemCELoss)
LOSS_REGISTRY.register(SmoothCELoss)
LOSS_REGISTRY.register(DiceLoss)
LOSS_REGISTRY.register(FocalTverskyLoss)
LOSS_REGISTRY.register(STCNLoss)
LOSS_REGISTRY.register(STCNLossV2)
LOSS_REGISTRY.register(ABL)
