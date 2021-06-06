import torch.nn as nn
import torch
import numpy as np

ignore_thresh = 0.5

# 均方损失函数
class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        pos_id = (targets==1.0).float()
        neg_id = (targets==0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss

class MSEWithLogitsLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        inputs = torch.sigmoid(logits)

        pos_id = (mask==1.0).float()
        neg_id = (mask==0.0).float()

        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        loss = 5.0*pos_loss + 1.0*neg_loss

        if self.reduction == 'mean':
            batch_size = logits.size(0)
            loss = torch.sum(loss) / batch_size

            return loss

        else:
            return loss

def compute_iou(anchor_boxes, gt_box):
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # xmin
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # ymin
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # xmax
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # ymax
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]

    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)
    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2  # xmin
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2  # ymin
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2  # xmax
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2  # ymin
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab

    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])

    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U

    return IoU

def set_anchors(anchor_size):
    anchor_number = len(anchor_size)
    anchor_boxes = np.zeros([anchor_number, 4])
    for index, size in enumerate(anchor_size):
        anchor_w, anchor_h = size
        anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])

    return anchor_boxes


def generate_txtytwth(gt_label, w, h, s, all_anchor_size):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # compute the center, width and height
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1. or box_h < 1.:
        # print('A dirty data !!!')
        return False

        # map the center, width and height to the feature map size
    c_x_s = c_x / s
    c_y_s = c_y / s
    box_ws = box_w / s
    box_hs = box_h / s

    # the grid cell location
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # generate anchor boxes
    anchor_boxes = set_anchors(all_anchor_size)
    gt_box = np.array([[0, 0, box_ws, box_hs]])
    # compute the IoU
    iou = compute_iou(anchor_boxes, gt_box)
    # We consider those anchor boxes whose IoU is more than ignore thresh,
    iou_mask = (iou > ignore_thresh)

    result = []
    if iou_mask.sum() == 0:
        # We assign the anchor box with highest IoU score.
        index = np.argmax(iou)
        p_w, p_h = all_anchor_size[index]
        tx = c_x_s - grid_x
        ty = c_y_s - grid_y
        tw = np.log(box_ws / p_w)
        th = np.log(box_hs / p_h)
        weight = 2.0 - (box_w / w) * (box_h / h)

        result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])

        return result

    else:
        # There are more than one anchor boxes whose IoU are higher than ignore thresh.
        # But we only assign only one anchor box whose IoU is the best(objectness target is 1) and ignore other
        # anchor boxes whose(we set their objectness as -1 which means we will ignore them during computing obj loss )
        # iou_ = iou * iou_mask

        # We get the index of the best IoU
        best_index = np.argmax(iou)
        for index, iou_m in enumerate(iou_mask):
            if iou_m:
                if index == best_index:
                    p_w, p_h = all_anchor_size[index]
                    tx = c_x_s - grid_x
                    ty = c_y_s - grid_y
                    tw = np.log(box_ws / p_w)
                    th = np.log(box_hs / p_h)
                    weight = 2.0 - (box_w / w) * (box_h / h)

                    result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])
                else:
                    # we ignore other anchor boxes even if their iou scores all higher than ignore thresh
                    result.append([index, grid_x, grid_y, 0., 0., 0., 0., -1.0, 0., 0., 0., 0.])

        return result


    # 损失函数


def loss(pred_conf, pred_cls, pred_txtytwth, label):
    obj = 5.0
    noobj = 1.0

    # create loss_f
    conf_loss_function = MSELoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')  # 交叉熵损失函数
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    pred_conf = torch.sigmoid(pred_conf[:, :, 0])
    pred_cls = pred_cls.permute(0, 2, 1)
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]

    gt_obj = label[:, :, 0].float()
    gt_cls = label[:, :, 1].long()
    gt_txtytwth = label[:, :, 2:-1].float()
    gt_box_scale_weight = label[:, :, -1]

    # objectness loss
    pos_loss, neg_loss = conf_loss_function(pred_conf, gt_obj)
    conf_loss = obj * pos_loss + noobj * neg_loss

    # class loss
    cls_loss = torch.mean(torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_obj, 1))

    # box loss
    txty_loss = torch.mean(
        torch.sum(torch.sum(txty_loss_function(pred_txty, gt_txtytwth[:, :, :2]), 2) * gt_box_scale_weight * gt_obj, 1))
    twth_loss = torch.mean(
        torch.sum(torch.sum(twth_loss_function(pred_twth, gt_txtytwth[:, :, 2:]), 2) * gt_box_scale_weight * gt_obj, 1))

    txtytwth_loss = txty_loss + twth_loss

    total_loss = conf_loss + cls_loss + txtytwth_loss

    return conf_loss, cls_loss, txtytwth_loss, total_loss

# 训练标签编码
def generate_dxdywh(get_label, w, h, s):
    xmin, ymin, xmax, ymax = get_label[:-1]
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax -xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1. or box_h < 1. :
        return  False

    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    #
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w)
    th = np.log(box_h)
    weight = 2.0 - (box_w / w)*(box_h / h) # 尺寸越大这个值越小，平衡小尺寸

    return grid_x, grid_y, tx, ty, tw, weight

# 训练标签编码
def gt_creator(input_size, stride, label_lists, anchor_size):
    """
    Input:
        input_size : list -> the size of image in the training stage.
        stride : int or list -> the downSample of the CNN, such as 32, 64 and so on.
        label_list : list -> [[[xmin, ymin, xmax, ymax, cls_ind], ... ], [[xmin, ymin, xmax, ymax, cls_ind], ... ]],
                        and len(label_list) = batch_size;
                            len(label_list[i]) = the number of class instance in a image;
                            (xmin, ymin, xmax, ymax) : the coords of a bbox whose valus is between 0 and 1;
                            cls_ind : the corresponding class label.
    Output:
        gt_tensor : ndarray -> shape = [batch_size, anchor_number, 1+1+4, grid_cell number ]
    """
    assert len(input_size) > 0 and len(label_lists) > 0
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    h = w = input_size

    # We  make gt labels by anchor-free method and anchor-based method.
    ws = w // stride
    hs = h // stride
    s = stride

    # We use anchor boxes to build training target.
    all_anchor_size = anchor_size
    anchor_number = len(all_anchor_size)

    gt_tensor = np.zeros([batch_size, hs, ws, anchor_number, 1 + 1 + 4 + 1 + 4])

    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            gt_class = int(gt_label[-1])
            results = generate_txtytwth(gt_label, w, h, s, all_anchor_size)
            if results:
                for result in results:
                    index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax = result
                    if weight > 0.:
                        if grid_y < gt_tensor.shape[1] and grid_x < gt_tensor.shape[2]:
                            gt_tensor[batch_index, grid_y, grid_x, index, 0] = 1.0
                            gt_tensor[batch_index, grid_y, grid_x, index, 1] = gt_class
                            gt_tensor[batch_index, grid_y, grid_x, index, 2:6] = np.array([tx, ty, tw, th])
                            gt_tensor[batch_index, grid_y, grid_x, index, 6] = weight
                            gt_tensor[batch_index, grid_y, grid_x, index, 7:] = np.array([xmin, ymin, xmax, ymax])
                    else:
                        gt_tensor[batch_index, grid_y, grid_x, index, 0] = -1.0
                        gt_tensor[batch_index, grid_y, grid_x, index, 6] = -1.0

    gt_tensor = gt_tensor.reshape(batch_size, hs * ws * anchor_number, 1 + 1 + 4 + 1 + 4)

    return gt_tensor

# v3编码
def multi_gt_creator(input_size, strides, label_lists=[], archor_size=None):
    batch_size = len(label_lists)
    h, w = input_size
    num_scale = len(strides)
    gt_tensor = []

    all_anchor_size = archor_size
    anchor_number = len(all_anchor_size) // num_scale
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h//s, w//s, anchor_number, 1+1+4+1+4]))
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            gt_class = int(gt_label[-1])
            xmin, ymin, xmax, ymax = gt_label[:-1]
            c_x = (xmax + xmin) / 2 * w
            c_y = (ymax + ymin) / 2 * h
            box_w = (xmax - xmin) * w
            box_h = (ymax - ymin) * h

            if box_w < 1. or box_h < 1.:
                continue

            anchor_boxes = set_anchors(all_anchor_size)
            gt_box = np.array([[0, 0, box_w, box_h]])
            iou = compute_iou(anchor_boxes, gt_box)

            iou_mask = (iou > ignore_thresh)

            if iou_mask.sum() == 0:
                index = np.argmax(iou)
                s_index = index // anchor_number
                ab_ind = index - s_index * anchor_number
                s = strides[s_index]
                p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                #
                c_x_s = c_x / s
                c_y_s = c_y / s
                grid_x = int(c_x_s)
                grid_y = int(c_y_s)
                #
                tx = c_x_s - grid_x
                ty = c_y_s - grid_y
                tw = np.log(box_w / p_w)
                th = np.log(box_h / p_h)
                weight = 2.0 - (box_w / w)*(box_h /h)

                if grid_y < gt_tensor[s_index].shape[1] and grid_x < gt_tensor[s_index].shape[2]:
                    gt_tensor[s_index][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                    gt_tensor[s_index][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
                    gt_tensor[s_index][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[s_index][batch_index, grid_y, grid_x, ab_ind, 6] = weight
                    gt_tensor[s_index][batch_index, grid_y, grid_x, ab_ind, 7:] = np.array([xmin, ymin, xmax, ymax])
            else:
                best_index = np.argmax(iou)
                for index, iou_m in enumerate(iou_mask):
                    if iou_m:
                        if index == best_index:
                            # 和上面相同
                        else:
                            s_index = index // anchor_number
                            ab_ind = index - s_index * anchor_number
                            s = strides[s_index]

    gt_tensor = [gt.reshape(batch_size, -1, 1+1+4+1+4) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, 1)

    return gt_tensor


def DIou(bboxes_a, bboxes_b, batch_size):
    B = batch_size

    x1234 = torch.cat([ bboxes_a[:, [0,2]], bboxes_b[:, [0,2]]], dim=1)
    y1234 = torch.cat([ bboxes_a[:, [1,3]], bboxes_b[:, [1,3]]], dim=1)

    #
    C = torch.sqrt((torch.max(x1234, dim=1)[0] - torch.min(x1234, dim=1)[0])**2 + \
                   (torch.max(y1234, dim=1)[0] - torch.min(y1234, dim=1)[0])**2)

    #
    points_1_x = (bboxes_a[:, 0] + bboxes_a[:, 2]) / 2.
    points_1_y = (bboxes_a[:, 1] + bboxes_a[:, 3]) / 2.

    points_2_x = (bboxes_b[:, 0] + bboxes_b[:, 2]) / 2.
    points_2_y = (bboxes_b[:, 1] + bboxes_b[:, 3]) / 2.
    D = torch.sqrt((points_2_x - points_1_x) ** 2 + (points_2_y - points_1_y) ** 2)
    #
    iou = Iou(bboxes_a, bboxes_b, B).view(-1)

    lens = D**2 / (C**2 + 1e-20)
    diou =iou -lens
    diou = diou.view(B, -1, 1)

    return diou








