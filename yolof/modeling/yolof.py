import copy
import logging
import numpy as  np
from typing import Dict, List, Tuple
import torch
from fvcore.nn import sigmoid_focal_loss_jit, giou_loss
from torch import Tensor, nn
import torch.distributed as dist
from torchvision.ops.boxes import box_iou

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.layers import batched_nms, cat, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage

from .encoder import DilatedEncoder
from .encoder import Decoder
from .box_regression import YOLOBox2BoxTransform
from .uniform_matcher import UniformMatcher

__all__ = ["YOLOF"]

@META_ARCH_REGISTRY.register()
class YOLOF(nn.Module):

    @configurable
    def __init__(
            self,
            *,
            backbone,
            encoder,
            decoder,
            anchor_generator,
            box2box_transform,
            anchor_matcher,
            num_classes,
            backbone_level="res5",
            pos_ignore_thresh=0.15,
            neg_ignore_thresh=0.7,
            focal_loss_alpha=0.25,
            focal_loss_gamma=2.0,
            box_reg_loss_type="giou",
            test_score_thresh=0.05,
            test_topk_candidates=1000,
            test_nms_thresh=0.6,
            max_detections_per_image=100,
            pixel_mean,
            pixel_std,
            vis_period=0,
            input_format="BGR"
    ):
        super().__init__()

        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

        # Anchors
        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher

        self.num_classes = num_classes
        self.backbone_level = backbone_level
        # Ignore thresholds:
        self.pos_ignore_thresh = pos_ignore_thresh
        self.neg_ignore_thresh = neg_ignore_thresh
        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.box_reg_loss_type = box_reg_loss_type
        assert self.box_reg_loss_type == 'giou', "Only support GIoU Loss."
        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format








