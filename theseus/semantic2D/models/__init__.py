from theseus.base.models import MODEL_REGISTRY

from .wrapper import ModelWithLoss

from .stcn import STCNModel
from theseus.semantic2D.models.segmodels import BaseSegModel
from theseus.semantic2D.models.swin.swin_unet import SwinUnet
from theseus.semantic2D.models.transunet import TransUnet


MODEL_REGISTRY.register(STCNModel)
MODEL_REGISTRY.register(BaseSegModel)
MODEL_REGISTRY.register(SwinUnet)
MODEL_REGISTRY.register(TransUnet)