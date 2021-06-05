from data import *

import numpy as np
import random

from data import BaseTransform, VOC_ROOT, VOCDetection
import numpy as np
import random
import argparse

class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def iou(box1, box2):
    x1, y1, w1, h1 = box1.x, box1.y, box1.w, box1.h
    x2, y2, w2, h2 = box2.x, box1.y, box1.w, box1.h

    S_1 = w1 * h1
    S_2 = w2 * h2

    xmin_1, ymin_1 = x1 - w1 / 2, y1 - h1/2
    xmax_1, ymax_1 = x1 + w1 / 2, y1 + h1/2
    xmin_2, ymin_2 = x2 - w2 / 2, y2 - h2 / 2
    xmax_2, ymax_2 = x2 + w2 / 2, y2 + h2 / 2

    I_w = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
    I_h = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)
    if I_w < 0 or I_h < 0:
        return 0
    I = I_w * I_h

    IoU = I / (S_1 + S_2 - I)

    return IoU

def init_centroids(boxes, n_anchors):
    centroids = []
    bboxes_num = len(boxes)

    centroids_index = int(np.random.choice(bboxes_num, 1)[0])
    centroids.append(boxes[centroids_index])
    print(centroids[0].w, centroids[0].h)

    for centroids_index in range(0, n_anchors-1):
        sum_distance = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - iou(box, centroids))
                if distance < min_distance:
                    min_distance = distance
            sum_distance +=min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance * np.random.random()

        for i in range(0, bboxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, box[i].h)
                break

    return centroids


def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []

    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - iou(box, centroids))
            if distance < min_distance:
                min_distance = distance
                group_index =centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= max(len(group_index[i], 1))
        new_centroids[i].h /= max(len((group_index[i]), 1))

    return new_centroids, groups, loss




def anchor_box_kmeans(total_gt_boxes, n_anchors, loss_convergence, iters, plus=True):

    boxes = total_gt_boxes
    centroids = []
    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:
        total_indexs = range(len(boxes))
        sample_indexs = random.sample(total_indexs, n_anchors)
        for i in sample_indexs:
            centroids.append(boxes[i])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while(True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations += 1
        print("Loss = %f" % loss)
        if abs(old_loss - loss) < loss_convergence or iterations > iters:
            break
        old_loss = loss

    print("k-means reslut :")
    for centroid in centroids:
        print(round(centroid.w, 2), round(centroid.h, 2), "area:", round(centroid.w, 2) * round(centroid.h, 2))

    return centroids

if __name__ == "__main__":

    print("ok")







