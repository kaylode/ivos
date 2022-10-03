from theseus.base.metrics import METRIC_REGISTRY

from .nsd import NormalizedSurfaceDistance
from .miou import mIOU
from .dicecoeff import DiceScore

METRIC_REGISTRY.register(NormalizedSurfaceDistance)
METRIC_REGISTRY.register(DiceScore)
METRIC_REGISTRY.register(mIOU)