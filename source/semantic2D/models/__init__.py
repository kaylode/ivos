from theseus.semantic.models import MODEL_REGISTRY

from .wrapper import ModelWithLoss

from .stcn import STCNModel
from .swin.swin_unet import SwinUnet
from .transunet.transunet_pos import TransUnetPE


MODEL_REGISTRY.register(STCNModel)
MODEL_REGISTRY.register(SwinUnet)
MODEL_REGISTRY.register(TransUnetPE)