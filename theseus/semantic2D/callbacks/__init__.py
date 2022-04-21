from theseus.semantic2D.callbacks.visualize_callbacks import VisualizerCallbacks
from theseus.semantic2D.callbacks.stcn_callback import STCNCallbacks
from theseus.base.callbacks import CALLBACKS_REGISTRY

CALLBACKS_REGISTRY.register(STCNCallbacks)
CALLBACKS_REGISTRY.register(VisualizerCallbacks)