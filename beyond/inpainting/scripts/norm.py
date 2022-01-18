import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import cv2
import argparse
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', help='Path to ground truth data', type=str) 
    args = parser.parse_args()
    return args

def load_flist(flist):
        if isinstance(flist, list):
            return flist
        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))+list(glob(path_true + '/*.JPG'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))
    
path = args.data_path

pathDir = load_flist(path)
#pathDir = list(glob(path + '/*.jpg')) + list(glob(path + '/*.png'))+list(glob(path + '/*.JPG'))
 
R_channel = 0
G_channel = 0
B_channel = 0




def img_axis(img): 
    image_shape = img.shape
    height, width = img.shape[0:2]
    if len(image_shape) != 3:
        img = np.repeat(img[:,:,np.newaxis], 3, 2) 

    return img, height, width

for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = imread(os.path.join(path, filename)) / 255.0
    img, height, width = img_axis(img)
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])
 
num = len(pathDir) * height * width  
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num
 
R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = imread(os.path.join(path, filename)) / 255.0
    img, height, width = img_axis(img)
    R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
    G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
    B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)
 
R_var = np.sqrt(R_channel / num)
G_var = np.sqrt(G_channel / num)
B_var = np.sqrt(B_channel / num)
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))
