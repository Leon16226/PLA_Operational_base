import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

class ZeroPad(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        # zero padding
        if height > width:
            image_ = np.zeros([height, height, 3])
            delta_w = height - width
            left = delta_w // 2
            image_[:, left:left+width, :] = image
            offset = np.array([[left / height, 0., left / height, 0.]])
            scale = np.array([[width / height, 1., width / height, 1.]])

        if boxes is not None:
            boxes = boxes * offset + offset

        return image_, boxes, labels, scale, offset

