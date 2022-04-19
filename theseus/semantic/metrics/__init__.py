from theseus.base.metrics import METRIC_REGISTRY

from .flare22.dicecoeff import *
from .miou import *

METRIC_REGISTRY.register(FLAREMetrics)
METRIC_REGISTRY.register(mIOU)