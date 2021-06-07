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


















