import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2

import glob
from ntpath import basename
from imageio import imread
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.color import rgb2gray


def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', '--gt', help='Path to ground truth data', type=str)
    parser.add_argument('--output-path', '--o', help='Path to output data', type=str)
    parser.add_argument('--valrst-path', '--v', default='./', help='Path to save val result', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    if np.sum(img_true + img_test) == 0:
        return 1
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)


def load_flist(flist):
        if isinstance(flist, list):
            return flist
        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))+list(glob.glob(path_true + '/*.JPG'))
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

path_true = args.data_path
path_pred = args.output_path

psnr = []
ssim = []
mae = []
names = []
index = 1

files = load_flist(path_true)
# files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))+list(glob(path_true + '/*.JPG'))

def img_resize(img, height, width, centerCrop=True):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

        img = cv2.resize(img, dsize=(height, width))

        return img
    else: 
        img = cv2.resize(img, dsize=(height, width))

        return img

for fn in sorted(files):
    name = basename(str(fn))
    names.append(name)

    # img_gt = (imread(str(fn)) / 255.0).astype(np.float32)
    # pred_name = str(fn).split('.')[0] + '_f.' + str(fn).split('.')[1]
    pred_name = str(fn)
    
    img_gt = (imread(str(fn)) / 255.0).astype(np.float32)
    # print(img_gt)
    img_pred = (imread(path_pred + '/' + basename(pred_name) ) / 255.0).astype(np.float32)
    
    img_gt = img_resize(img_gt, 256, 256)
    # print(img_gt.shape)
    img_gt = rgb2gray(img_gt)
    img_pred = rgb2gray(img_pred)

    if args.debug != 0:
        plt.subplot('121')
        plt.imshow(img_gt)
        plt.title('Groud truth')
        plt.subplot('122')
        plt.imshow(img_pred)
        plt.title('Output')
        plt.show()

    psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
    ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=51))
    mae.append(compare_mae(img_gt, img_pred))
    if np.mod(index, 100) == 0:
        print(
            str(index) + ' images processed',
            "PSNR: %.4f" % round(np.mean(psnr), 4),
            "SSIM: %.4f" % round(np.mean(ssim), 4),
            "MAE: %.4f" % round(np.mean(mae), 4),
        )
    index += 1

np.savez(args.output_path + '/metrics.npz', psnr=psnr, ssim=ssim, mae=mae, names=names)
print(
    "PSNR: %.4f" % round(np.mean(psnr), 4),
    "PSNR Variance: %.4f" % round(np.var(psnr), 4),
    "SSIM: %.4f" % round(np.mean(ssim), 4),
    "SSIM Variance: %.4f" % round(np.var(ssim), 4),
    "MAE: %.4f" % round(np.mean(mae), 4),
    "MAE Variance: %.4f" % round(np.var(mae), 4)
)
