import torch
import cv2
import numpy as np

def detection_collate(batch):
    target = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        target.append(torch.FloatTensor(sample[1]))





def base_transform(image, size, mean, std):
    x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
    x /= 255.
    x -= mean
    x /= std
    return x


class BaseTransform:
    def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean, self.std), boxes, labels

if __name__ == '__main__':
    img = torch.randint(0, 255, (3, 2, 2))
    img = img.type(torch.float32)
    img = img.unsqueeze(0)

    img = torch.nn.functional.interpolate(img, size=[4, 4], mode='bilinear')

    print(img)

