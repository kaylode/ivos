from theseus.base.models import MODEL_REGISTRY

from .wrapper import ModelWithLoss

from .stcn import STCNModel
from theseus.semantic2D.models.segmodels import BaseSegModel

MODEL_REGISTRY.register(STCNModel)
MODEL_REGISTRY.register(BaseSegModel)
