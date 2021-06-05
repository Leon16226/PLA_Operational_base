from __future__ import absolute_import
from __future__ import division
import torch as t
import numpy as np
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
from torchvision.ops import nms
# from model.utils.nms import non_maximum_suppression

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from utils.config import opt


class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean=(0., 0., 0., 0.,),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset("evaluate")

    # @property 只读属性
    # class.n_class 直接就可以访问
    @property
    def n_class(self):
        return self.head.n_class

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, rois_indices, anchor = \
            self.rpn(h, img_size, scale)
        roi_cls_locs, rois_scores = self.head(
            h, rois, rois_indices)
        return roi_cls_locs, rois_scores, rois, rois_indices

    def use_preset(self, preset):
        # 预测的时候使用
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('xxx')

    def predict(self, imgs, sizes=None,visualize=False):
        # model.train() 启用BathNormalization和Droput
        # model.eval()  不启用
        # 测试事前要加上，否则只要输入数据即使不训练，model也会改变权值
        self.eval()
        if visualize:
            self.use_preset('')
            prepared_imgs=list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs

        bboxes = list()
        labels = list()
        scores = list()
        for img,size in zip(prepared_imgs,sizes):
            ...

        self.use_preset("evaluate")
        self.train()
        return bboxes, labels, scores



