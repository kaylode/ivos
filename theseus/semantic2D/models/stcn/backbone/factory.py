from theseus.semantic2D.models.stcn.backbone.mod_resnet import ResNetBackbone

def create_model(name, pretrained:bool = True, **kwargs):

    if 'resnet' in name:
        return ResNetBackbone(name, pretrained, **kwargs)