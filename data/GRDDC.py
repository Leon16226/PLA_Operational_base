import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random
import shutil

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

Data_Root = "/Volumes/ud/train"
Copy_Images_Root = "/Volumes/ud/D40Images"
Country_Names = {"Czech": 3594, "India": 9891, "Japan": 13132}
Need_Object_Name = "D40"

if __name__ == "__main__":

    imagepath_all = []

    for i, country in enumerate(Country_Names):
        imagepath_one_country = []
        for n in range(0, Country_Names[country] + 1):
            # 获得xml文件序号
            file_order = 1000000 + n
            file_order = str(file_order)[1::]
            annopath = osp.join('%s', "%s", "annotations", "xmls", "%s.xml")
            annopath = annopath % (Data_Root, country, country + '_' + file_order)
            if (osp.exists(annopath)):
                target = ET.parse(annopath).getroot()
                for obj in target.iter('object'):
                    # difficult = int(obj.find("difficult").text) == 1
                    # if difficult:
                    #     continue
                    name = obj.find("name").text.lower().strip()
                    # 如果需要的话拿出图片路径
                    if name == Need_Object_Name:
                        filename = target.find("filename").text.lower().strip()
                        imagepath_one_country.append(filename)
                    print("寻找图片中...：%i/%i" % (n, Country_Names[country] + 1))
        print('%s寻找图片完毕' % country)
        imagepath_all.append(imagepath_one_country)

# 复制图片
for i in range(0, 3):
    for path in imagepath_all[i]:
        imagpath = osp.join('%s', "%s", "images", "%s")
        country = "Czech"
        if i == 1:
            country = "India"
        elif i == 2:
            country = "Japan"
        imagpath = imagpath % (Data_Root, country, path)
        shutil.copy(imagpath, Copy_Images_Root + '/' + path)
    print('%s复制图片完毕' % country)
