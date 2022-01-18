import cv2
import os
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='example.jpg', help='path to the dataset')
args = parser.parse_args()

img = cv2.imread(args.path)
dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst[dst < dst.mean()+15] = 0
print(dst.mean(), np.median(dst))

# dst = cv2.GaussianBlur(dst, (3,3), 0)

dst = cv2.equalizeHist(dst)

cv2.imwrite('res.png',dst)
