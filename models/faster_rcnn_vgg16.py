from __future__ import absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt


class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        super().__init__()
        self.clssifier = classifier
        self.clas_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.clas_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        # 把RoIs尺度不一的bbox归一化到相同大小
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        roi_indices = at.totensor(roi_indices).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]

        # is_contiguous Tensor底层一维数组元素的存储顺序和其按行优先一维展开的元素顺序是否一致
        # Tensor的有些操作会让数组在内存不相邻，也不连续
        # 所以需要contiguous

        # 举例 t = torch.arange(12).reshape(3,4)
        # t.flatten() 可以看到是行优先
        # 行优先和列优先的存储结构不同
        # 不论是行优先，还是列优先，访问矩阵中的下一个元素都是通过偏移来实现，行优先存储模式下1偏移，列优先是3偏移

        # 为什么需要contiguous？
        # torch.view等方法操作需要连续的Tensor

        # 只有很少几个操作不改变tensor本身内容
        # 而是而改变Tensor的元数据(比如stride...)
        # view() expand() transpose()
        # 被上述函数使用过后Tensor布局就不同了
        # 需要调用contiguous()函数重新拷贝一份

        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.clssifier(pool)
        roi_cls_locs = self.clas_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def decom_vgg16():
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain))
    else:
        model = vgg16(not opt.load_path)

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_frop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class FasterRCNNVGG16(FasterRCNN):
    feat_stride = 16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]):
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512,512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride
        )

        head = VGG16RoIHead(
            n_class=n_fg_class+1,
            roi_size=7,
            spatial_scale=(1./self.feat_stride),
            classifier=classifier
        )
        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
