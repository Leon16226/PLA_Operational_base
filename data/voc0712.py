import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random

from torch.utils.data.dataset import T_co

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

VOC_ROOT =  "/Users/apple/ProjectsPytorch/ZDATA/"


class VOCAnnotationTransform(object):

    def __init__(self, class_to_ind=None, keep_difficult=False):
        super().__init__()
        # 比如 人-0 猫-1
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES)))
        )
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = []
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = cur_pt / width if i % i == 0 else cur_pt /height
                bbox.append(cur_pt)
            label_inx = self.class_to_ind[name]
            bndbox.append(label_inx)
            res += [bndbox]

        return  res



class VOCDetection(data.Dataset):
    def __init__(self, root, img_size,
                 image_sets=[('2007', 'trainval'),('2012', 'trainval')],
                 transform = None, target_tansform=VOCAnnotationTransform(),
                 dataset_name='VOC0712', mosaic=False) :
        super().__init__()
        self.root = root
        self.img_size = image_sets
        self.transform = transform
        self.target_transform = target_tansform
        self.name = dataset_name
        self.mosaic = mosaic

        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s,jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))



    def __getitem__(self, index) :
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        # 转变为可用坐标
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        # 基本数据增强
        if self.transform is not None:
            # check labels
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)

            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

if __name__ == '__main__':

    # 定义基本的数据增强
    def base_transform(image, size, mean):
        x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x

    class BaseTransform:
        def __init__(self, size, mean):
            self.size = size
            self.mean = np.array(mean, dtype=np.float32)

        def __call__(self, image, boxes=None, labels=None):
            return base_transform(image, self.size, self.mean), boxes, labels

    # 训练尺寸
    img_size = 640

    dataset = VOCDetection(VOC_ROOT, img_size,
                          transform=BaseTransform([img_size, img_size], (0, 0, 0)),
                          )


