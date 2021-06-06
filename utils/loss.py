import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel

def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps

class BCEBlurWithLogitsLoss(nn.Module):

    def __init__(self, alpha=0.05):
        super().__init__()
        # 就是Sigmoid-BCELoss结合
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

