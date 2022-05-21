from theseus.semantic2D.models.stcn.backbone.mod_resnet import ResNetBackbone
from theseus.semantic2D.models.stcn.backbone.mod_mbv3 import MobileNetBackbone

def create_model(name, pretrained:bool = True, **kwargs):

    if 'resnet' in name:
        return ResNetBackbone(name, pretrained, **kwargs)

    if 'mbv3' in name:
        return MobileNetBackbone(name, pretrained=pretrained,**kwargs)