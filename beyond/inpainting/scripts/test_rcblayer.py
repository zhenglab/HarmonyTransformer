import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.functional as FF
from PIL import Image

from imageio import imread
import random
import numpy as np
import os


def random_crop(npdata, crop_size):
    
    height, width = npdata.shape[0:2]
    mask = np.ones((height, width))

    # h = random.randint(0, height - crop_size)
    # w = random.randint(0, width - crop_size)
    h = 32
    w = 32

    mask[h: h+crop_size, w: w+crop_size] = 0
    crop_image = npdata[h: h+crop_size, w: w+crop_size]
    
    return crop_image, (w, h), mask

def same_padding(images, ksizes, strides, rates):   
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images

def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x

def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x

def extract_image_patches(images, ksizes, strides, padding='same'):

    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, [1, 1])
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
            Only "same" or "valid" are supported.'.format(padding))
    batch_size, channel, height, width = images.size()

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                            padding=0,
                            stride=strides)
    patches = unfold(images)
    return patches


def compose_bad(good, fm, kernel_size=[2,2], k_h=2, k_w=2): #good (8,64,32,32) ; bad (8,64,64,64)
        # good tenser b,c,w,h ; offset b,c,w,h 
        (b, c, h, w) = fm.shape
        patch_bad_all = extract_image_patches(fm, ksizes=kernel_size, strides=kernel_size, padding='same').view(b, c, k_h, k_w, -1).permute(0, 4, 1, 2, 3)
        patch_final_groups = torch.split(torch.zeros(patch_bad_all.shape), 1, dim=0)

        for i in range(c):
            cc = 1
            fm_ = fm[:,i].unsqueeze(1)
            good_ = good[:,i].unsqueeze(1)

            patch_bad = extract_image_patches(fm_, ksizes=kernel_size, strides=kernel_size, padding='same').view(b, 1, k_h, k_w, -1).permute(0, 4, 1, 2, 3)
            
            patch_bad_groups = torch.split(patch_bad, 1, dim=0)
            
            patch_good = extract_image_patches(good_, ksizes=kernel_size, strides=kernel_size, padding='same').view(b, 1, k_h, k_w, -1).permute(0, 4, 1, 2, 3)
            patch_good_groups = torch.split(patch_good, 1, dim=0)
            
            fold_1_2 = nn.Fold(output_size=[h//2, w//2],kernel_size=kernel_size, stride=kernel_size)

            for p_g, p_b, p_f in zip(patch_good_groups, patch_bad_groups, patch_final_groups):
                # print(g.shape, p_g.shape, p_b.shape, p_f.shape)
                bad_patch = p_b[0]  # [1024, 64, 2, 2]
                good_patch = p_g[0]
                final_patch = p_f[0]

                escape_NaN = torch.FloatTensor([1e-4])
                good_max = torch.max(torch.sqrt(reduce_sum(torch.pow(good_patch, 2), axis=[1, 2, 3], keepdim=True)), escape_NaN)
                good_normed = good_patch / good_max

                g = good_normed.permute(1, 2, 3, 0).view(1, cc*k_h*k_w, -1) 
                g = fold_1_2(g)           

                yi = FF.conv2d(g, bad_patch, stride=2)
                yi = yi.view(1, yi.shape[1], -1)   # 1, 32*32, 1024

                offset = torch.argmax(yi, dim=2, keepdim=True).squeeze()  # 1, 1024, 1          

                for o in range(offset.shape[0]):
                    index = offset[o]
                    val = good_patch[index]
                    final_patch[o][i] = val       # 1024*64*2*2

        finals = torch.cat(patch_final_groups, dim=0).permute(0, 2, 3, 4, 1).view(b, c*k_h*k_w, -1)
        fold = nn.Fold(output_size=[h, w],kernel_size=kernel_size, stride=kernel_size)
        output = fold(finals)
        return output


path = './img/1.jpg'
files = []

data = imread(path)
data_good, pos, mask = random_crop(data, 128)

fm = F.to_tensor(Image.fromarray(data)).float().view(1, 3, 256, 256)
good = F.to_tensor(Image.fromarray(data_good)).float().view(1, 3, 128, 128)
mask = F.to_tensor(Image.fromarray(mask)).float().view(1, 256, 256)

final = compose_bad(good, fm)

# final = final * mask + fm * (1-mask)
final = final * 255.0
final = final.permute(0, 2, 3, 1)
final = final.int()[0] 

im = Image.fromarray(final.numpy().astype(np.uint8).squeeze())
im.save('0.jpg')