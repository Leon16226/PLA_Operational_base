import glob
import hashlib
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

# 如果是旋转过的宽，高就要调换
def exif_size(img):
    s = img.size # (w, h)
    try:
        rotation = dict(img._gettextif().items())[orientation]
        if rotation == 6:
            s = (s[1], s[0])
        elif rotation == 8:
            s = (s[1], s[0])
    except:
        pass

    return s



class LoadImageAndLabels(Dataset):

    def __init__(self, path, img_size=416, batch_size=16, augment=False, hyp=None, rect=False,image_weight=False,
                 cache_images=False, single_cls=False, pad=0.0) :
        super().__init__()
        try:
            path = str(Path(path))
            parent = str(Path(path).parent) + os.sep
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    f = f.read().splitlines()
                    f = [x.replace('./', parent) if x.startswith('./') else x for x in f]
            elif os.path.isdir(path):
                f = glob.iglob(path + os.sep + '*.*')
            else:
                raise Exception('%s does not exist'% path)
            # 不用平台用的路径分隔符不同
            self.img_files = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats]
        except:
            raise Exception('Error loding')

        n = len(self.img_files)
        assert n > 0 ,'No images found'
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)
        nb = bi[-1] + 1

        self.n = n # 图片数量
        self.batch = bi # 批次数量
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.imge_weights = image_weight
        self.rect = False if image_weight else rect
        self.mosaic = self.augment and not self.rect

        self.label_files = [x.replace('images','labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]

        sp = path.replace('.txt', '') + '.shapes'
        try:
            with open(sp, 'r') as f:
                s = [x.split() for x in f.read().splitlines()]
                assert  len(s) == n , "c"
        except:
            # tqdm是进度条的意思
            s = [exif_size(Image.open(f) for f in tqdm(self.img_files, desc='Reading image shapes'))]
            # %f 浮点数 %g 指数或浮点数
            np.savetxt(sp, s, fmt='%g')

        self.shapes = np.array(s, dtype=np.float64)

        if self.rect:
            s = self.shapes
            ar = s[:, 1] / s[:, 0]
            irect = ar.arsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]

            shapes = [[1,1]] * nb
            for i in range(nb):
                # 找出对应批次的高宽比
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int) * 32

            # Cache labels
            # weighted CE 训练需要这个步骤
            self.imgs = [None] * n
            self.labels = [np.zero((0,5), dtype=np.float32)] * n
            create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
            nm, nf, ne, ns, nd = 0, 0, 0, 0, 0 # number missing,found,datasubset,duplicate
            np_labels_path = str(Path(self.label_files[0]).parent) + '.npy'
            if os.path.isfile(np_labels_path):
                s = np_labels_path
                x = np.load(np_labels_path, allow_pickle=True)
                if len(x) == n:
                    self.labels = x
                    labels_loaded = True
            else:
                s = path.replace('images', 'labels')

            pbar = tqdm(self.label_files)
            for i, file in enumerate(pbar):
                if labels_loaded:
                    l = self.labels[i]
                else:
                    try:
                        with open(file, 'r') as f:
                            l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                    except:
                        nm +=1
                        continue

                if l.shape[0]:
                    assert l.shape[1] == 5, 'x'
                    assert (l>=0).all(), 'x'
                    assert (l[:, 1:] <=1).all(),'x'
                    if np.unique(l, axis=0).shape[0] < l.shape[0]:
                        nd += 1
                    if single_cls:
                        l[:, 0] = 0
                    self.labels[i] = l
                    nf += 1

                    # 创建一个小型的数据集进行试验
                    if create_datasubset and ns < 1E4:
                        if ns ==0:
                            create_folder(path='./datasubset')
                            os.makedirs('./datasubset/images')
                        exculde_classes = 43
                        if exculde_classes not in l[:, 0]:
                            ns += 1
                        with open('./datasubet/images.txt','a') as f:
                            f.write(self.img_files[i] + '\n')

                    # 为两阶段分类器提取目标检测的检测框
                    # 默认开关是关的
                    if extract_bounding_boxes:
                        p = Path(self.img_files[i])
                        img = cv2.imread(str(p))
                        h, w = img.shape[:2]
                        for j, x in enumerate(l):
                            f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                            if not os.path.exists(Path(f).parent):
                                os.makedirs(Path(f).parent)

                            b = x[1:] * [w, h, w, h]
                            b[2:] = b[2:].max()
                            b[2:] = b[2:] * 1.3 + 30
                            b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                            b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
                            b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                            assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'x'

                else:
                    ne += 1 # 没图片

                pbar.desc = 'Cacheing label %s'% s
            assert nf > 0 or n == 20288, "no labels"
            if not labels_loaded and n > 1000:
                np.save(np_labels_path, self.labels)

            # Cache imaegs
            if cache_images:
                gb = 0 # Gigabytes of cached images GB为单位
                pbar = tqdm(range(len(self.img_files)), desc='Caching images')
                self.img_hw0, self.img_hw = [None] * n,[None] * n
                for i in pbar:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)
                    gb += self.imgs[i].nbytes
                    pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

            # 删除损坏的文件 (手动设置)
            detect_corrupted_images = False

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.image_weights:
             index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            img, labels = load_mosaic(self, index)
            shapes = None
        else:
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = []
            x = self.labels[index]
            if x.size > 0:
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            if not self.mosaic:
                # img, labels = random_affine

        # augment_hsv()

        nL = len(labels)
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            labels[:, [2, 4]] /= img.shape[0]
            labels[:, [1, 3]] /= img.shape[1]

        if self.augment:
            pass

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes








def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y





def letterbox(img, new_shape=(416,416), color=(114, 114, 114), auto=True, scaleFill=F, scaleup=True):
    #
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r
    new_uppad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_uppad[0], new_shape[0] - new_uppad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_uppad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_shape:
        img = cv2.resize(img, new_uppad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)











def load_image(self, index):
    img = self.imgs[index]
    if img is None:
        path = self.img_files[index]
        img = cv2.imread(path)
        assert img is not None, "x"
        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]


def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y



def create_folder(path='./new_folder'):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

























