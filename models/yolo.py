import torch
import torch.nn as nn
from backbone import resnet18
import numpy as np
import tools


class myYOLO(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 hr=False):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        # 下采样倍数
        self.stride = 32
        # input_size [h,w] 训练时候resize的宽和高 比如416 x 416
        self.grid_cell = self.create_grid(input_size)
        self.input_size = input_size
        # 相对大小
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=device).float()

        # 网络结构
        self.backbone = resnet18(pretrained=True)
        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)

    # 获取网格左上角坐标
    def create_grid(self, input_size):
        w, h = input_size[1], input_size[0]
        ws, hs = w // self.stride, h // self.stride
        # 生成网格
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        # 从（7，7，2）变成 （1，7x7，2）
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 2).to(self.device)

        return grid_x

    # 更新grid
    def set_grid(self, input_size):
        self.input_size = input_size
        # (1,7x7,2)
        self.grid_cell = self.create_grid(input_size)
        # (1,1,4)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

    # 解码坐标
    def decode_boxes(self, pred):
        """
        input box :  [tx, ty, tw, th]
        output box : [xmin, ymin, xmax, ymax]
        """
        output = torch.zeros_like(pred)
        pred[:, :, :2] = torch.sigmoid(pred[:, :, :2]) + self.grid_cell
        pred[:, :, 2:] = torch.exp(pred[:, :, 2:])

        # [c_x, c_y, w, h] -> [xmin, ymin, xmax, ymax]
        output[:, :, 0] = pred[:, :, 0] * self.stride - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] * self.stride - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] * self.stride + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] * self.stride + pred[:, :, 3] / 2

        return output

    def nms(self, dets, scores):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        # 从小到大排列输出索引
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def diou_nms(self, dets, scores):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []


def postprocess(self, all_local, all_conf, im_shape=None):
    bbox_pred = all_local
    prob_pred = all_conf

    cls_ins = np.argmax(prob_pred, axis=1)
    prob_pred = prob_pred[(np.arange(prob_pred.sape[0], cls_ins))]
    scores = prob_pred.copy()

    # threshold
    keep = np.where(scores >= self.conf_thresh)
    bbox_pred = bbox_pred[keep]
    scores = scores[keep]
    cls_ins = cls_ins[keep]

    # NMS

    keep = np.zeros(len(bbox_pred), dtype=np.int)

    for i in range(self.num_classes):
        inds = np.where(cls_ins == i)[0]


def forward(self, x, target=None):
    # backbone
    _, _, C_5 = self.backbone(x)

    # pred
    prediction = self.pred(C_5)

    # 把[B, C, H, W]的预测变成了[B, HxW, C]
    prediction = prediction.view(C_5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)
    B, HW, C = prediction.size()

    # [B, H*W, 1]
    conf_pred = prediction[:, :, :1]
    # [B, H*W, num_cls]
    cls_pred = prediction[:, :, 1: 1 + self.num_classes]
    # [B, H*W, 4]
    txtytwth_pred = prediction[:, :, 1 + self.num_classes:]

    # test
    if not self.trainable:
        with torch.no_grad():
            # bacth size = 1
            all_conf = torch.sigmoid(conf_pred)[0]
            # 所有值要在0~1之间
            all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)

            all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_conf)

            # separate box pred and class conf
            all_conf = all_conf.to('cpu').numpy()
            all_class = all_class.to('cpu').numpy()
            all_bbox = all_bbox.to('cpu').numpy()

            bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)

            return bboxes, scores, cls_inds
    else:
        conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss()

        return conf_loss, cls_loss, txtytwth_loss, total_loss


if __name__ == '__main__':
    a = list([2, 3, 4])
    a = np.array(a)
    a = a + 3
    print(a)
