import glob
import math
import os
import random
import shutil
import subprocess
import time
from copy import copy
from pathlib import Path
from sys import platform

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from . import torch_utils  # , google_utils


def labels_to_class_weights(labels, nc=80):
    if labels[0] is None:
        return torch.Tensor()

    labels = np.concatenate(labels, 0)
    classes = labels[:, 0].astype(np.int)
    weights = np.bincount(classes, minlength=nc)


# 为了让生成的随机数相同
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def check_git_status():
    if platform in ['linux', 'darwin']:
        # Suggest 'git pull' if repo is out of date
        s = subprocess.check_output('if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
        if 'Your branch is behind' in s:
            print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')


def check_file(file):
    # Searches for file if not found locally
    if os.path.isfile(file):
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found


# old
def build_taegets_old(model, targets):

    nt = len(targets)

    tcls, tbox, indices, av = [], [], [], []

    # 是否多尺度训练
    multi_gpu = type(model) in (nn.parallel.DataParallel,
                                nn.parallel.DistributedDataParallel)

    reject, use_all_anchors = True, True
    for i in model.yolo_layers:
        # ng 代表num of grid (13,13) anchor_vec [[x,y],[x,y]]
        if multi_gpu:
            ng = model.module.module_list[i].ng
            anchor_vec = model.module.module_list[i].anchor_vec
        else:
            ng = model.module_list[i].ng,
            anchor_vec = model.module_list[i].anchor_vec

        t, a = targets, []

        gwh = t[:, 4:6] * ng[0] # 变成相对尺寸

        if nt:
            iou = wh_iou(anchor_vec, gwh) # 计算iou

            if use_all_anchors:
                na = len(anchor_vec)
                a = torch.arange(na).view(
                    (-1, 1)).repeat([1, nt]).view(-1)
                # a = [0,0,1,1,2,2]
                t = targets.repeat([na, 1])
                # [2,6]->[6,6]
                gwh = gwh.repeat([na, 1])
            else:
                iou, a = iou.max(0)

            if reject:
                j = iou.view(-1) > model.hyp['iou_t']
                t, a, gwh = t[j], a[j], gwh[j]

        b, c = t[:,:2].long().t()

        gxy = t[:, 2:4] * ng[0]

        gi, gj = gxy.long().t()

        indices.append((b, a, gj, gi))

        gxy -=gxy.floor()

        tbox.append(torch.cat((gxy,gwh),1))

        av.append(anchor_vec[a])

        tcls.append(c)

    return tcls, tbox, indices, av

# 适应度函数：判断其中一代的好坏
# 适应度高，代表这一代性能好
# Precision,Recall,mAP，F1
def fitness(x):
    w = [0.0, 0.01, 0.99, 0.00] # 加权
    return (x[:, :4] * w).sum(1)
























def build_targets(p, targets, model):
    nt = targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()

    style = None
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for i, j in enumerate(model.yolo_layers):
        # anchor_vec = 实际值 / 32
        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
        na = anchors.shape[0]
        at = torch.arange(na).view(na, 1).repeat(1, nt)

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if nt:
            j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']
            a, t = at[j], t.repeat(na, 1, 1)[j]


def compute_loss(p, targets, model):
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchors = build_targets(p, targets, model)
    red = 'mean'

    #
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)

    cp, cn = smooth_BCE(eps=0.0)

    nt = 0
    for i, pi in enumerate(p):
        b, a, gj, gi = indices[i]
        tobj = torch.zeros_like(pi[..., 0])

        nb = b.shape[0]
        if nb:









