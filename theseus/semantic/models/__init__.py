from theseus.base.models import MODEL_REGISTRY

from .wrapper import ModelWithLoss

from .stcn import STCNModel

MODEL_REGISTRY.register(STCNModel)
