from theseus.semantic2D.models import MODEL_REGISTRY
from .cps import CrossPseudoSupervision


MODEL_REGISTRY.register(CrossPseudoSupervision)