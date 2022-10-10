import torch
import torch.nn as nn
import torch.nn.functional as F
from source.semantic2D.models.stcn.utilities.loading import load_pretrained_model 
from theseus.utilities.loading import load_state_dict

__all__ = ['MobileNetV3', 'mobilenetv3']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetBackbone(nn.Module):
    def __init__(self, mode='mbv3s', width_mult=1.0, extra_chan=0, pretrained=True):
        super(MobileNetBackbone, self).__init__()
        input_channel = 16
        last_channel = 1280
        
        if mode == 'mbv3l':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]

            self.f16_dim = 80
            self.f8_dim = 24
            self.f4_dim =  16
            last_bneck = [1, 3, 10, 0]

        elif mode == 'mbv3s':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]

            self.f16_dim = 48
            self.f8_dim = 24
            self.f4_dim =  16
            last_bneck = [1, 3, 8, 0]
        else:
            raise NotImplementedError

        # building first layer
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel

        self.layers_os4 = [conv_bn(1+extra_chan, input_channel, 2, nlin_layer=Hswish)]#after 1st bottle neck
        self.layers_os8 = [] #after 3rd bottle neck
        self.layers_os16 = [] #after 8th bottle neck
        self.layers_os32 = [] #last
        layers = [self.layers_os4, self.layers_os8, self.layers_os16, self.layers_os32]
        
        n_layer = 0
        layer = layers[n_layer]
        n_last_bneck = last_bneck[n_layer]
        # building mobile blocks
        for i, (k, exp, c, se, nl, s) in enumerate(mobile_setting):
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            layer.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

            if i+1 == n_last_bneck:
                n_layer +=1
                layer = layers[n_layer]
                n_last_bneck = last_bneck[n_layer]

        # make it nn.Sequential
        self.layers_os4, self.layers_os8, self.layers_os16, self.layers_os32 = [nn.Sequential(*layer) for layer in layers]
        
        if pretrained:
            self.load_pretrained(mode)
        else:
            self._initialize_weights()

    def forward(self, x, return_more=False):
        x1 = self.layers_os4(x)
        x2 = self.layers_os8(x1)
        x3 = self.layers_os16(x2)
        # x4 = self.layers_os32(x3)
        if return_more:
            return x3,x2,x1
        else:
            return x3

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_pretrained(self, name):
        pretrained_path = load_pretrained_model(name)
        state_dict = torch.load(pretrained_path, map_location='cpu')
        load_state_dict(self, state_dict, strict=False)