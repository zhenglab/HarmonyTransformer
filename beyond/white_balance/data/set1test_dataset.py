"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os.path
from os.path import join
from os import listdir
from glob import glob
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as transforms
from util import util
from scipy.io import loadmat
import random

class Set1testDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths = []
        self.isTrain = opt.isTrain
        self.image_size = opt.crop_size
        max_trdata=12000
        self.patch_size = 128
        self.patch_num_per_image = 4
        random.seed(1000)
        if opt.isTrain==True:
            print('loading training file')
            self.imgfiles = []
            # fold = 1
            # tfolds = list(set([1, 2, 3]) - set([fold]))
            # # logging.info(f'Training process will use {max_trdata} training images randomly selected from folds {tfolds}')
            # files = loadmat(join('data/folds', 'fold%d_.mat' % fold))
            # files = files['training']
            # self.imgfiles = []
            # # logging.info('Loading training images information...')
            # for i in range(len(files)):
            #     temp_files = glob(opt.dataset_root + files[i][0][0])
            #     for file in temp_files:
            #         parts = file.split('_')
            #         base_name = ''
            #         for i in range(len(parts) - 2):
            #             base_name = base_name + parts[i] + '_'
            #         if os.path.isfile(base_name + 'S_AS.jpg') and os.path.isfile(base_name + 'T_AS.jpg'):
            #             self.imgfiles.append(file)
            # if max_trdata is not 0 and len(self.imgfiles) > max_trdata:
            #     random.shuffle(self.imgfiles)
            #     self.imgfiles = self.imgfiles[0:max_trdata]

            # with open("train.txt","w") as f:
            #     for v in self.imgfiles:
            #         f.write(v)
            #         f.write("\n")
            # logging.info(f'Creating dataset with {len(self.imgfiles)} examples')
            self.trainfile = 'data/train.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.imgfiles.append(line.rstrip().replace("set_1_input", 'set_1_input_png').replace(".jpg", '.png'))
        elif opt.isTrain==False:
            # self.imgfiles = [join(opt.dataset_root, file) for file in listdir(opt.dataset_root)
            #             if file.endswith('.JPG')]
            # self.imgfiles = []
            # fold = 1
            # files = loadmat(join('data/folds', 'fold%d_.mat' % fold))
            # files = files['validation']
            # # logging.info('Loading training images information...')
            # for i in range(len(files)):
            #     temp_files = glob(opt.dataset_root + files[i][0][0])
            #     for file in temp_files:
            #         self.imgfiles.append(file)
            
            # with open("st_1_test.txt","w") as f:
            #     for v in self.imgfiles:
            #         f.write(v)
            #         f.write("\n")
            # assert 0
            self.imgfiles = []
            self.trainfile = 'data/set_1_test.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.imgfiles.append(line.rstrip())

        # get the image paths of your dataset;
          # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ]
        self.transforms = transforms.Compose(transform_list)
        print(len(self.imgfiles))
        # assert 1==0
    def __getitem__(self, index):
        if self.isTrain:
            gt_ext = ('G_AS.png', 'T_AS.png', 'S_AS.png')
            # gt_ext = ('G_AS.png', 'T_AS.jpg', 'S_AS.jpg')
            img_file = self.imgfiles[index]

            in_img = Image.open(img_file)
            # get image size
            w, h = in_img.size
            # get ground truth images
            parts = img_file.split('_')
            base_name = ''
            for i in range(len(parts) - 2):
                base_name = base_name + parts[i] + '_'
            gt_awb_file = base_name + gt_ext[0]
            awb_img = Image.open(gt_awb_file.replace("_input_png/", "_gt/"))
            gt_t_file = base_name + gt_ext[1]
            t_img = Image.open(gt_t_file)
            gt_s_file = base_name + gt_ext[2]
            s_img = Image.open(gt_s_file)
            # get flipping option
            flip_op = np.random.randint(3)
            # get random patch coord
            patch_x = np.random.randint(0, high=w - self.patch_size)
            patch_y = np.random.randint(0, high=h - self.patch_size)
            in_img_patches = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
            awb_img_patches = self.preprocess(awb_img, self.patch_size, (patch_x, patch_y), flip_op)
            img_t_patches = self.preprocess(t_img, self.patch_size, (patch_x, patch_y), flip_op)
            img_s_patches = self.preprocess(s_img, self.patch_size, (patch_x, patch_y), flip_op)
            for j in range(self.patch_num_per_image - 1):
                # get flipping option
                flip_op = np.random.randint(3)
                # get random patch coord
                patch_x = np.random.randint(0, high=w - self.patch_size)
                patch_y = np.random.randint(0, high=h - self.patch_size)
                temp = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
                in_img_patches = np.append(in_img_patches, temp, axis=0)
                temp = self.preprocess(awb_img, self.patch_size, (patch_x, patch_y), flip_op)
                awb_img_patches = np.append(awb_img_patches, temp, axis=0)
                temp = self.preprocess(t_img, self.patch_size, (patch_x, patch_y), flip_op)
                img_t_patches = np.append(img_t_patches, temp, axis=0)
                temp = self.preprocess(s_img, self.patch_size, (patch_x, patch_y), flip_op)
                img_s_patches = np.append(img_s_patches, temp, axis=0)
            return {'image': torch.from_numpy(in_img_patches), 'gt-AWB': torch.from_numpy(awb_img_patches),
                    'gt-T': torch.from_numpy(img_t_patches), 'gt-S': torch.from_numpy(img_s_patches), 'img_path':img_file}
        else:
            s = 656
            img_file = self.imgfiles[index]
            image = Image.open(img_file)
            # image_resized = image.resize((round(image.width / max(image.size) * s), round(image.height / max(image.size) * s)))
            image_resized = image.resize([128,128])
            w, h = image_resized.size
            if w % 2 ** 4 == 0:
                new_size_w = w
            else:
                new_size_w = w + 2 ** 4 - w % 2 ** 4

            if h % 2 ** 4 == 0:
                new_size_h = h
            else:
                new_size_h = h + 2 ** 4 - h % 2 ** 4

            inSz = (new_size_w, new_size_h)
            if not ((w, h) == inSz):
                image_resized = image_resized.resize(inSz)

            image = np.array(image)
            image_resized = np.array(image_resized)
            img = image_resized.transpose((2, 0, 1))
            img = img / 255
            img = torch.from_numpy(img)
            return {'image': img, 'img_path':img_file}




        path = self.image_paths[index]
        name_parts=path.split('_')
        mask_path = self.image_paths[index].replace('composite_images','masks')
        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
        target_path = self.image_paths[index].replace('composite_images','real_images')
        target_path = target_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')

        # comp = util.retry_load_images(path)
        # mask = util.retry_load_images(mask_path)
        # real = util.retry_load_images(target_path)

        comp = Image.open(path).convert('RGB')
        real = Image.open(target_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')

        if np.random.rand() > 0.5 and self.isTrain:
            comp, mask, real = tf.hflip(comp), tf.hflip(mask), tf.hflip(real)

        if comp.size[0] != self.image_size:
            # assert 0
            comp = tf.resize(comp, [self.image_size, self.image_size])
            mask = tf.resize(mask, [self.image_size, self.image_size])
            real = tf.resize(real, [self.image_size,self.image_size])
        
        comp = self.transforms(comp)
        mask = tf.to_tensor(mask)
        # mask = 1-mask
        real = self.transforms(real)

        # comp = real
        # mask = torch.zeros_like(mask)
        # inputs=torch.cat([real,mask],0)
        inputs=torch.cat([comp,mask],0)
        
        return {'inputs': inputs, 'comp': comp, 'real': real,'img_path':path,'mask':mask}

    def __len__(self):
        """Return the total number of images."""
        return len(self.imgfiles)

    @classmethod
    def preprocess(cls, pil_img, patch_size, patch_coords, flip_op):
        if flip_op is 1:
            pil_img = ImageOps.mirror(pil_img)
        elif flip_op is 2:
            pil_img = ImageOps.flip(pil_img)

        img_nd = np.array(pil_img)
        assert len(img_nd.shape) == 3, 'Training/validation images should be 3 channels colored images'
        img_nd = img_nd[patch_coords[1]:patch_coords[1]+patch_size, patch_coords[0]:patch_coords[0]+patch_size, :]
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans / 255

        return img_trans

