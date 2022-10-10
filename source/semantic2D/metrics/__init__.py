from theseus.semantic.metrics import METRIC_REGISTRY

from .nsd import NormalizedSurfaceDistance

METRIC_REGISTRY.register(NormalizedSurfaceDistance)