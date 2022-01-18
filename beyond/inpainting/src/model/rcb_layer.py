import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from torch.autograd import Variable
from ..utils import extract_image_patches, same_padding, \
                    reduce_mean, reduce_sum

class RCBLayer(nn.Module):
    def __init__(self, re_nc, x_nc):
        super(RCBLayer, self).__init__()

        self._11_1 = nn.Sequential(
            nn.Conv2d(re_nc, 1, kernel_size=1),
            nn.ReLU()
        )

        self._11_2 = nn.Sequential(
            nn.Conv2d(x_nc, 1, kernel_size=1),
            nn.ReLU()
        )

    def compose_bad(self, recon, reference): #good (8,64,32,32) ; bad (8,64,64,64)
        # good tenser b,c,w,h ; offset b,c,w,h 
        (b, c, h, w) = reference.shape

        patch_bad = extract_image_patches(reference, ksizes=[2,2], strides=[2,2], padding='same').view(b, c, 2, 2, -1).permute(0, 4, 1, 2, 3)
        patch_bad_groups = torch.split(patch_bad, 1, dim=0)
        
        good_groups = torch.split(recon, 1, dim=0)
        patch_good = extract_image_patches(recon, ksizes=[2,2], strides=[2,2], padding='same').view(b, c, 2, 2, -1).permute(0, 4, 1, 2, 3)
        patch_good_groups = torch.split(patch_good, 1, dim=0)

        patch_final_groups = torch.split(torch.zeros(patch_bad.shape).cuda(), 1, dim=0)

        # finals = []
         
        # g => good, b => bad
        # good_groups => 8 * (64,32,32)
        # patch_bad_groups => 8 * (1024,64,2,2)
        # patch_final_groups => 8 * (1024,64,2,2)
        for g, p_g, p_b, p_f in zip(good_groups, patch_good_groups, patch_bad_groups, patch_final_groups):
            p_b = p_b[0]  # [1024, 64, 2, 2]
            p_g = p_g[0]

            # escape_NaN = torch.FloatTensor([1e-4]).cuda()
            # max_bad = torch.max(torch.sqrt(reduce_sum(torch.pow(p_b, 2), axis=[1, 2, 3], keepdim=True)), escape_NaN)
            # bad_normed = p_b / max_bad    # bad_normed 1024, 64, 2, 2 
            bad_normed = p_b

            gg = same_padding(g, [2, 2], [1, 1], [1, 1])
            yi = F.conv2d(gg, bad_normed, stride=1)  # 1, 1024, 32, 32
            yi = yi.view(1, yi.shape[1], -1)   # 1, 32*32, 1024

            offset = torch.argmax(yi, dim=2, keepdim=True)  # 1, 1024, 1            
            for o in range(offset.shape[1]):
                index = offset[:,o,:][0][0] // 4
                # print(index)
                val = p_g[index]
                p_f[0][o] = val       # 1024*64*2*2
            # finals.append(p_f) 

        # print(patch_final_groups[0][0][0][0])
        finals = torch.cat(patch_final_groups, dim=0).permute(0, 2, 3, 4, 1).view(b, c*2*2, -1)
        # print(finals.shape)
        # final = finals.contiguous().view(int_final)
        fold = nn.Fold(output_size=[h, w],kernel_size=[2,2], stride=[2,2])
        output = fold(finals)
        return output
    
    # postion => (h, w); good_size => (h, w)
    def forward(self, spade_fm, x):        
        # print(spade_fm.shape, x.shape)
        
        recon = self._11_1(spade_fm) 
        reference = self._11_2(x) 

        x = self.compose_bad(recon, reference)

        return x