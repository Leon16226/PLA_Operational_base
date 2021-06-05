#!/home/dai/anaconda3/bin/python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile
import numpy as np

# 根据自己的数据标签修改
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
Images_Save_Root = ''
Annota_Save_Root = ''

Image_Get_Root = ''
Annota_Get_Root = ''

Train_Name_Root = ''
Val_Name_Root = ''


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# 输入id转变annotation，并且存储
def convert_annotation(image_id, sub_path='/train'):
    in_file = open(Annota_Get_Root + '/%s.xml' % image_id)
    out_file = open(Annota_Save_Root + sub_path + '/%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()


# 转移图片
def transf_images(image_id, sub_path='/train'):
    image_path = Image_Get_Root + '/%s.jpg' % image_id
    copyfile(image_path, Images_Save_Root + sub_path)


if __name__ == "__main__":
    with open(Train_Name_Root, 'r') as f:
        print("开始读取文件")
        one_item_array = {}
        lines = f.readlines()
        convert_annotation(np.ndarray(lines), '/train')
        transf_images(np.ndarray(lines), '/train')
    f.close()

    with open(Val_Name_Root, 'r') as f:
        print("开始读取文件")
        one_item_array = {}
        lines = f.readlines()
        convert_annotation(np.ndarray(lines), '/val')
        transf_images(np.ndarray(lines), '/val')
    f.close()
