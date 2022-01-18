import os 
import glob
import scipy
import torch
import random
import numpy as np
import cv2
import torchvision.transforms.functional as F

from torch.utils.data import DataLoader
# from PIL import Image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from imageio import imread
from .utils import random_crop, center_crop, side_crop 
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .masks import Masks
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, input_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.input_size = config.INPUT_SIZE
        self.center = config.CENTER
        self.model = config.MODEL
        self.augment = augment
        self.training = training
        self.data = self.load_flist(input_flist)
        self.side = config.SIDE
        self.mean = config.MEAN
        self.std = config.STD
        self.count = 0
        self.pos = None
        self.batchsize = config.BATCH_SIZE
        self.catmask = config.CATMASK
        self.datatype = config.DATATYPE
        self.mask_index = 0
        if self.datatype == 2:
            self.scence_width = 512
            self.scence_height = 256
        self.known_mask = False
        if self.known_mask:
            self.mask_file = self.load_flist(mask_flist)
        if self.training == False:
            self.test_mask = self.load_flist(mask_flist)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # print(self.data[index])
        item = self.load_item(index)

        return item

    def resize(self, img, width, height):
        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)

        return img

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)
        # return os.path.basename(name).split('.')[0]

    def load_mask_name(self, index):
        name = self.test_mask[self.mask_index-1]
        return os.path.basename(name)
    
    def load_item(self, index):
        #size = self.input_size
        # print(self.data[index])
        data = imread(self.data[index])

        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
            data = data.repeat(3, axis=2)
        if self.datatype == 1:
            data = self.resize(data, self.input_size, self.input_size)
        if self.datatype == 2:
            data = self.resize(data, self.scence_width, self.scence_height)
        
        if self.training:
            if self.count == 0:
                self.mask = self.mask_generate(self.input_size, self.input_size)
        else:
            mask_index = random.randint(0, len(self.test_mask) - 1)
            # self.mask = imread(self.test_mask[mask_index])
            self.mask = imread(self.test_mask[index])
            # print(self.test_mask[index])
            self.mask = cv2.resize(self.mask, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        self.count += 1
        if self.count == self.batchsize:
            self.count = 0
            
        fmask_data = self.mask

        if self.augment and np.random.binomial(1, 0.5) > 0:
            data = data[:, ::-1, ...]
            fmask_data = fmask_data[:, ::-1, ...]
        
        # data = self.to_tensor(data)
        data_norm = self.to_tensor_norm(data)
        data = self.to_tensor(data)
        input_data = data * (1 - self.to_tensor(fmask_data))
        # input_data = data_norm * (1 - self.to_tensor(fmask_data))
        mask_data = data * (1 - self.to_tensor(fmask_data))
        
        return data, input_data, self.to_tensor(fmask_data), mask_data

    def mask_generate(self, h, w):
        # mask_re_o = self.re_o_mask(h, w)
        mask_irr_in = Masks.get_random_mask(h, w)
        # mask_re_in = self.re_in_mask(h, w)
        # index = random.randint(0, len(self.mask_file) - 1)
        # mask_irr_o = imread(self.mask_file[index])
        # print(mask_irr_in)
        return mask_irr_in
        # return random.choice([mask_re_o, mask_irr_in, mask_irr_o])
        
    def re_o_mask(self, h, w):
        mask = np.ones((h, w))
        crop_size=h//2

        h_offset = random.randint(0, h - crop_size)
        w_offset = random.randint(0, w - crop_size)

        mask[h_offset: h_offset + crop_size, w_offset: w_offset + crop_size] = 0

        return mask

    def re_in_mask(self, h, w):
        mask = np.zeros((h, w))
        crop_size=h//2

        h_offset = random.randint(0, h - crop_size)
        w_offset = random.randint(0, w - crop_size)

        mask[h_offset: h_offset + crop_size, w_offset: w_offset + crop_size] = 1

        return mask

    def img_resize(self, img, width, height, centerCrop=False):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)

        return img

    def locate_mask(self, data, mask):
        height, width = data.shape[0:2]
        coord = 0
        for i in range(width):
            for j in range(height):
                if (mask[i][j] != 0):
                    coord = (j,i)
                    break
            if (mask[i][j] != 0):
                break
        inner_img = data[i:i+128, j:j+128]
        return inner_img, coord
    
    def dealimage(self, data, mask):
        rc, pos = self.locate_mask(data, mask)
        return rc, pos

    # def cpimage(self, data):
    #     rc, pos, mask = random_crop(data, int(data.shape[1]/2), self.datatype)
    #     return rc, pos, mask
    def cpimage(self, data):
        if self.known_mask:
            # print(" seq: ",self.seq," mask file: ",self.mask_file[self.seq])
            mask = imread(self.mask_file[self.seq])
            rc, pos = self.dealimage(data, mask)
            self.pos = pos
        rc, pos, mask = random_crop(data, int(data.shape[1]/2), self.datatype, self.count, self.pos, self.known_mask)
        # rc, pos, mask = center_crop(data, int(data.shape[1]/2))
        self.pos = pos
        return rc, pos, mask
    
    def gray_fmap(self, fmap_data):
        fmap_data = cv2.cvtColor(fmap_data, cv2.COLOR_BGR2GRAY)
        fmap_data[fmap_data < fmap_data.mean()+15] = 0
        fmap_data = cv2.equalizeHist(fmap_data)
        
        return fmap_data


    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        if isinstance(flist, str):
            # print(flist)
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                # try:
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                # except:
                    # print(11, flist)
                #    return [flist]
        
        return []

    def load_test_flist(self, mask_flist_arr):
        file_dirs = []
        # print(lenmask_flist_arr)
        for i in range(len(mask_flist_arr)):
            file_dirs.append(np.genfromtxt(mask_flist_arr[i], dtype=np.str, encoding='utf-8'))
        return file_dirs
    
    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t
    
    def to_tensor_norm(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        img_t = F.normalize(img_t, self.mean, self.std)  # 输入mean 和 std
        return img_t


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item