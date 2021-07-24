from typing import Type

import numpy as np
import torch
from torchvision.ops import nms
from models.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox


# 制作target用于Proposal的训练
# （讲20000多个候选的anchor中选出256个）
class AnchorTargetCreator(object):

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        super().__init__()
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

    def _create_label(self, inside_index, anchor, bbox):
        # 全部标号置为-1
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_iou, max_ious, gt_argmax_iou = \
            self._calc_ious(anchor, bbox, inside_index)

        # <0.3负样本
        label[max_ious < self.neg_iou_thresh] = 0

        # 最匹配置为正样本
        label[gt_argmax_iou] = 1

        # >0.7正样本
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample postive labels
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disabel_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=True)
            label[disabel_index] = -1

        # subsample negative labels
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disabel_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disabel_index] = -1

        return argmax_iou, label

    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxed
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        #
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


def _get_inside_index(anchor, H, W):
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


# 生成ROIs = region of interest
# train     : (20000 -> 12000 -> nms -> 2000)
# inference : (20000 -> 6000 -> nms -> 300) 提高速度
class ProposalCreator:

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_szie=16):
        super().__init__()
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_szie

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        roi = loc2bbox(anchor, loc)

        # clip predicted boxes to image
        # RoIs全部剪裁到图片的区域内
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2), 0, img_size[0]])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        # min_size = 16  scale = 1.
        # 计算图片宽、高小于一定大小的直接去掉
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = [keep]

        # ->12000
        # 保留属于前景的概率 -> 排序后，保留前12000个
        order = score.ravel().argsor()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # nms -> 2000 RiOs
        # 抑制重复的框
        keep = nms(
            torch.from_numpy(roi).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        return roi


# 制作target为后续网络训练
# 2000 RoIs -> 128RoIs
# 输入RoIHead，对rois进行分类，同时位置微调

# RoIs : 2000x4
class ProposalTargetCreator(object):

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
        super().__init__()
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox, _ = bbox.shape

        roi = np.concatenate((roi, bbox), axis=0)
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)  # 这么多的被采样为前景
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        gt_roi_label = label[gt_assignment] + 1

        # 选0.25的foreground
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # 选中剩下的作为背景
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label
