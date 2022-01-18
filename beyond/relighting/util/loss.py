import torch
import torch.nn.functional as F
import numpy as np


C1 = 0.01**2
C2 = 0.03**2
def global_ssim(feat, fg_mask, alpa=1, beta=1, gamma=1):
    N, C, H, W = feat.size()
    fg_mask_sum = torch.sum(fg_mask.view(N, 1, -1), dim=2, keepdim=True)
    bg_mask_sum = H*W - fg_mask_sum+1e-8
    fg_mask_sum = fg_mask_sum+1e-8

    bg = feat*(1-fg_mask)
    fg = feat*fg_mask

    # avg pooling to 1*1
    bg_pooling = torch.sum(bg.view(N,C,-1), dim=2, keepdim=True).div(bg_mask_sum) #each channel mean  == channel avg pooling
    fg_pooling = torch.sum(fg.view(N,C,-1), dim=2, keepdim=True).div(fg_mask_sum)

    bg_distribution_mu = torch.mean(bg_pooling, dim=1, keepdim=True)  #all channel mean
    fg_distribution_mu = torch.mean(fg_pooling, dim=1, keepdim=True)  #all channel mean

    bg_distribution_std = (torch.var(bg_pooling, dim=1, keepdim=True) + 1e-8).sqrt()
    fg_distribution_std = (torch.var(fg_pooling, dim=1, keepdim=True) + 1e-8).sqrt()


    # luminance = (2*bg_distribution_mu*fg_distribution_mu+C1).div(bg_distribution_mu.pow(2)+fg_distribution_mu.pow(2)+C1)
    # contrast = (2*bg_distribution_std*fg_distribution_std+C2).div(bg_distribution_std.pow(2)+fg_distribution_std.pow(2)+C2)
    
    # fg_bg_gobal_conv = torch.matmul((bg_pooling-bg_distribution_mu).permute(0,2,1), (fg_pooling-fg_distribution_mu))/(C-1)
    # structure = (fg_bg_gobal_conv+C2/2).div(bg_distribution_std*fg_distribution_std+C2/2)
    
    # global_co = luminance.pow(alpa)*contrast.pow(beta)*structure.pow(gamma)
    # global_co = global_co.squeeze(-1)
    fg_bg_gobal_conv = torch.matmul((bg_pooling-bg_distribution_mu).permute(0,2,1), (fg_pooling-fg_distribution_mu))/(C-1)
    fg_bg_r = fg_bg_gobal_conv.div(bg_distribution_std*fg_distribution_std+1e-8)
    fg_bg_r = fg_bg_r.squeeze(-1)
    global_co = fg_bg_r.abs()
    return global_co