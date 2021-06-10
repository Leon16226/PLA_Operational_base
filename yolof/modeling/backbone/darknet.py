import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec

from ..utils import get_norm

try:
    from mish_cuda import MishCuda as Mish
except Exception:
    logger = logging.getLogger(__name__)
    logger.warning("")

    def mish(x):
        return x.mul(F.softplus(x).tanh())

    class Mish(nn.Module):
        def __init__(self):
            super(Mish, self).__init__()

        def forward(self, x):
            return mish(x)

def ConvNormActivation(inplanes,
                       planes,
                       kernel_size=3,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=1,
                       norm_type="BN"):
    layers = []
    layers.append(nn.Conv2d(inplanes,
                            planes,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            bias=False))
    layers.append(get_norm(norm_type, planes, eps=1e-4, momentum=0.03))
    layers.append(Mish())
    return nn.Sequential(*layers)


class DarkBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 dilation=1,
                 downsample=None,
                 norm_type="BN"):
        super().__init__()

        self.downsample = downsample

        self.bn1 = get_norm(norm_type, inplanes, eps=1e-4, momentum=0.03)
        self.bn2 = get_norm(norm_type, planes, eps=1e-4, momentum=0.03)

        self.conv1 = nn.Conv2d(
            planes,
            inplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False
        )

        self.activation = Mish()

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out += identity

        return out

class CrossStagePartialBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stage_layers,
                 is_csp_first_stage,
                 dilation=1,
                 stride=2,
                 norm_type="BN"):
        super().__init__()

        self.base_layer = ConvNormActivation(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            norm_type=norm_type
        )
        self.partial_transition1 = ConvNormActivation(
            inplanes=planes,
            planes = inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type
        )
        self.stage_layers = stage_layers
        self.partial_transition2 = ConvNormActivation(
            inplanes=inplanes if not is_csp_first_stage else planes,
            planes=inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type
        )
        self.fuse_transition = ConvNormActivation(
            inplanes=planes if not is_csp_first_stage else planes * 2,
            planes=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type
        )

    def forward(self, x):
        x = self.base_layer(x)

        out1 = self.partial_transition1(x)

        out2 = self.stage_layers(x)
        out2 = self.partial_transition2(out2)

        out = torch.cat([out2, out1], dim=1)
        out = self.fuse_transition(out)

        return out



def make_dark_layer(block,
                    inplanes,
                    planes,
                    num_blocks,
                    dilation=1,
                    stride=2,
                    norm_type="BN"):
    downsample = ConvNormActivation(
        inplanes=inplanes,
        planes=planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        norm_type=norm_type
    )

    layers = []
    for i in range(0, num_blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                downsample=downsample if i == 0 else None,
                dilation=dilation,
                norm_type=norm_type
            )
        )

    return nn.Sequential(*layers)

def make_cspdark_layer(block,
                       inplanes,
                       planes,
                       num_blocks,
                       is_csp_first_stage,
                       dilation=1,
                       norm_type="BN"):
    downsample = ConvNormActivation(
        inplanes=planes,
        planes=planes if is_csp_first_stage else inplanes,
        kernel_size=1,
        stride=1,
        padding=0,
        norm_type=norm_type
    )

    layers = []
    for i in range(0, num_blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes if is_csp_first_stage else inplanes,
                downsample=downsample if i == 0 else None,
                dilation=dilation,
                norm_type=norm_type
            )
        )
        return nn.Sequential(*layers)

class DarkNet(Backbone):
    arch_settings = {
        53:(DarkBlock, (1, 2, 8, 8, 4))
    }

    def __init__(self,
                 depth,
                 with_csp=False,
                 out_fretures=["res5"],
                 norm_type="BN",
                 res5_dilation=1):
        super(DarkNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError("")
        self.with_csp = with_csp
        self._out_features = out_fretures
        self.norm_type = norm_type
        self.res5_dilation = res5_dilation

        self.block, self.stage_blocks = self.arch_settings[depth]
        self.inplanes = 32

        self._make_stem_layer()

        self.dark_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            planes = 64 * 2 ** i
            dilation = 1
            stride = 2
            if i == 4 and self.res5_dilation == 2:
                dilation = self.res5_dilation
                stride = 1
            if not self.with_csp:
                layer = make_dark_layer(
                    block=self.block,
                    inplanes=self.inplanes,
                    planes=planes,
                    num_blocks=num_blocks,
                    dilation=dilation,
                    stride=stride,
                    norm_type=self.norm_type
                )
            else:
                layer = make_dark_layer(
                    block=self.block,
                    inplanes=self.inplanes,
                    planes=planes,
                    num_blocks=num_blocks,
                    is_csp_first_stage=True if i==0 else False,
                    dilation=dilation,
                    norm_type=self.norm_type
                )
                # 从DarkBlock -> 变成 CrossStagePartialBlock
                # 一次次变的
                layer = CrossStagePartialBlock(
                    self.inplanes,
                    planes,
                    stage_layers=layer,
                    is_csp_first_stage=True if i==0 else False,
                    dilation=dilation,
                    stride=stride,
                    norm_type=self.norm_type
                )
            self.inplanes = planes
            # out -> layer5
            layer_name = 'layer{}'.format(i + 1)
            # 父类方法
            self.add_module(layer_name, layer)
            self.dark_layers.append(layer_name)

        # freeze stage<=2
        for p in self.conv1.parameters():
            p.requires_grad = False
        for p in self.bn1.parameters():
            p.requires_grad = False
        for p in self.layer1.parameters():
            p.requires_grad = False
        for p in self.layer2.parameters():
            p.requires_grad = False

    # 3 -> 32
    def _make_stem_layer(self):
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = get_norm(
            self.norm_type, self.inplanes, eps=1e-4, momentum=0.03
        )
        self.act1 = Mish()

    def forward(self, x):
        outputs = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        for i, layer_name in enumerate(self.dark_layers):
            layer = getattr(self, layer_name)
            x = layer(x)
        outputs[self._out_features[-1]] = x
        return outputs

    def output_shape(self):
        return {
            "res5": ShapeSpec(
                channels=1024, stride=16 if self.res5_dilation == 2 else 32
            )
        }

@BACKBONE_REGISTRY.register()
def build_darknet_backbone(cfg, input_shape=None):
    depth = cfg.MODEL.DARKNET.DEPTH
    with_csp = cfg.MODEL.DARKNET.WITH_CSP
    out_features = cfg.MODEL.DARKNET.OUT_FEATURES
    norm_type = cfg.MODEL.DARKNET.NORM
    res5_dialtion = cfg.MODEL.DARKNET.RES5_DILATION
    return DarkNet(
        depth=depth,
        with_csp=with_csp,
        out_features=out_features,
        norm_type=norm_type,
        res5_dilation=res5_dialtion
    )
































