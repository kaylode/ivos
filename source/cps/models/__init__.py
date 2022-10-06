from theseus.semantic.models import MODEL_REGISTRY
from .cps import CrossPseudoSupervision


MODEL_REGISTRY.register(CrossPseudoSupervision)