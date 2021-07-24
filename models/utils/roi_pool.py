import torch
from torch import nn, Tensor

from torch.nn.modules.utils import _pair
from torch.jit.annotations import List, BroadcastingList2

from torchvision.extension import _assert_has_ops
from ._utils import convert_boxes_to_roi_format, check_roi_boxes_shape

def roi_pool(
        input: Tensor,
        boxes: Tensor,
        output_size: BroadcastingList2[2], # int or Tuple[int, int]
        spatial_scale: float = 1.0
) -> Tensor:
    check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = _pair(output_size)
    #
    output, _ = torch.ops.torchvison.roi_pool(input, rois, spatial_scale,
                                              output_size[0], output_size[1])



