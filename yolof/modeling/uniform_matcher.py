import numpy as np
import torch
from torch import nn

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

class UniformMatcher(nn.Module):

    def __init__(self, match_times: int = 4):
        super().__init__()
        self.match_times = match_times

    @torch.no_grad
    def foward(self, pre_boxes, anchors, targets):
        bs, num_quries = pre_boxes.shape[:2]

        out_bbox = pre_boxes.flatten(0, 1)
        anchors = anchors.flatten(0, 1)

        tgt_bbox = torch.cat([v.get_boxes.tensor for v in targets])

        cost_bbox = torch.cdist(
            box_xyxy_to_cxcywh(out_bbox), box_xyxy_to_cxcywh(tgt_bbox), p=1)
        cost_bbox_anchors = torch.cdist(
            box_xyxy_to_cxcywh(anchors), box_xyxy_to_cxcywh(tgt_bbox), p=1)

        # Final cost matrix
        C = cost_bbox
        C = C.view(bs, num_quries, -1).cpu()
        C1 = cost_bbox_anchors
        C1 = C1.view(bs, num_quries, -1).cpu()









