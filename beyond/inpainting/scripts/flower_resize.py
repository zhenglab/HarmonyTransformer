import os
import numpy as np
from PIL import Image
import cv2

#pt = '/home/sqw/dataset/food-101/images/'
path = '/home/sqw/dataset/flower.flist'
output = '/home/ouc/dataset/102_Category_Flower_Dataset/centercrop-flowers/'

def img_resize(img, width, height, centerCrop=True):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # center 
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)

    return img

for i in range(8189):
    print(i)
    with open(path, 'r') as f:
        line_f = f.readlines() 
    
    line_f[i] = line_f[i][:-1] 

    a = line_f[i].split('/')[-1]

    img = cv2.imread(line_f[i]) 

    img_c = img_resize(img, 256, 256)

    img_path = os.path.join(output, a)

    cv2.imwrite(img_path, img_c) 
