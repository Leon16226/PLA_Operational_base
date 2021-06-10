from typing import List

from fvcore.nn import c2_xavier_fill
import torch
import torch.nn as nn

from detectron2.layers import ShapeSpec

from .utils import get_activation, get_norm

class DialedEncode(nn.Module):

    def __init__(self, cfg, input_shape:List[ShapeSpec]):
        super().__init__()
        self.backbone_level = cfg.MODEL.YOLOF.ENCODER.BACKBONE_LEVEL
        self.in_channels = cfg.MODEL.YOLOF.ENCODER.IN_CHANNELS
        self.encoder_channels = cfg.MODEL.YOLOF.ENCODER.NUM_CHANNELS
        self.block_mid_channels = cfg.MODEL.YOLOF.ENCODER.BLOCK_MID_CHANNELS
        self.num_residual_blocks = cfg.MODEL.YOLOF.ENCODER.NUM_RESIDUAL_BLOCKS
        self.block_dilations = cfg.MODEL.YOLOF.ENCODER.BLOCK_DILATIONS
        self.norm_type = cfg.MODEL.YOLOF.ENCODER.NORM
        self.act_type = cfg.MODEL.YOLOF.ENCODER.ACTIVATION

        # init
        self._init_layers()
        self._init_weight()

    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(self.in_channels,
                                      self.encoder_channels,
                                      kernel_size=1)
        self.lateral_norm = get_norm(self.norm_type, self.encoder_channels)
        self.fpn_conv = nn.Conv2d(self.encoder_channels,
                                  self.encoder_channels,
                                  kernel_size=3,
                                  padding=1)
        self.fpn_norm = get_norm(self.norm_type, self.encoder_channels)
        encoder_blocks = []
        # 一共8个瓶颈层
        for i in range(self.num_residual_blocks):
            dialation = self.block_dilations[i] # 逐渐增大
            encoder_blocks.append(
                Bottleneck(
                    self.encoder_channels,
                    self.block_mid_channels,
                    dialation=dialation,
                    norm_type=self.norm_type,
                    act_type=self.act_type
                )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def _init_weight(self):
        c2_xavier_fill(self.lateral_conv)
        c2_xavier_fill(self.fpn_conv)
        for m in [self.lateral_norm, self.fpn_conv]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        for m in self.dilated_encoder_blocks:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature: torch.Tensor):
        out = self.lateral_norm(self.lateral_conv(feature))
        out = self.fpn_norm(self.fpn_conv(out))
        return self.dilated_encoder_blocks(out)



class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: int = 128,
                 dilation: int =1,
                 norm_type: str = 'BN',
                 act_type: str = 'ReLU'
                 ):
        super().__init__()
        # 通过瓶颈层中间3x3卷积 进行dilation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels),
            get_norm(norm_type, mid_channels),
            get_activation(act_type)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            get_norm(norm_type, mid_channels),
            get_activation(act_type)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            get_norm(norm_type, in_channels),
            get_activation(act_type)
        )

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity # wh不变，特征图加起来
        return out
