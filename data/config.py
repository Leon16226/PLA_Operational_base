# from .config import *
# form data import *

IGNORE_THRESH = 0.5

# new yolo config
train_cfg = {
    'lr_epoch': (60, 90, 160),
    'max_epoch': 160,
    'min_difm': [416, 416]
}

# multi level anchor box config for VOC and COCO
# yolo_v3
MULTI_ANCHOR_SIZE = [[30.65, 39.12],   [50.3, 102.62],   [94.98, 64.55],
                     [93.5, 177.51],   [165.25, 113.85], [161.83, 240.95],
                     [304.64, 150.34], [251.28, 306.53], [369.38, 261.55]]

MULTI_ANCHOR_SIZE_COCO = [[11.89, 14.24],   [30.14, 35.62],   [45.99, 87.04],
                          [92.23, 44.43],   [130.78, 99.73],  [78.99, 170.81],
                          [290.39, 123.89], [165.27, 233.33], [332.57, 279.8]]


