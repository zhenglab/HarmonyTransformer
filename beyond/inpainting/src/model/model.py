import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as tdist
import torchvision.models as models
import imp

import numpy as np

from .networks import FCHTGenerator
# from .networks import SPADEGenerator, MultiscaleDiscriminator
# from .vae import VAE
from .loss import ColorLoss, PerceptualLoss, AdversarialLoss, KLDLoss, VGG16FeatureExtractor
# StyleLoss, FeatureAvgLoss, MRFLoss
from ..utils import template_match, Adam16

import math


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0
        self.device = config.DEVICE
        self.imagine_g_weights_path = os.path.join(config.PATH, 'g.pth')
    def load(self):
        if self.name == 'ImagineGAN':
            if os.path.exists(self.imagine_g_weights_path):
                print('Loading %s Model ...' % self.name)

                g_data = torch.load(self.imagine_g_weights_path)
                self.g.load_state_dict(g_data['params'])
                self.iteration = g_data['iteration']
        
    def save(self, ite):
        print('\nSaving %s...\n' % self.name)
        if self.name == 'ImagineGAN':
            # print(self.name == 'ImagineGAN')
            torch.save({
                'iteration': self.iteration,
                'params': self.g.state_dict()}, self.imagine_g_weights_path + '_' + str(ite))

class ImagineModel(BaseModel):
    def __init__(self, config):
        super(ImagineModel, self).__init__('ImagineGAN', config)
        if config.GEN_TYPE == 'fcht':
            g = FCHTGenerator(config)
        l1_loss = nn.L1Loss()
        color_loss = ColorLoss()
        content_loss = PerceptualLoss()
        
        self.add_module('g', g)
        self.add_module('l1_loss', l1_loss)
        self.add_module('color_loss', color_loss)
        self.add_module('content_loss', content_loss)

        self.lossNet = VGG16FeatureExtractor()
        
        # self.g_optimizer = Adam16(params=g.parameters(), lr=float(config.G_LR), betas=(config.BETA1, config.BETA2), weight_decay=0.0, eps=1e-8)
        self.g_optimizer = optim.Adam(params=g.parameters(), lr=float(config.G_LR), betas=(config.BETA1, config.BETA2))

    def set_position(self, pos, patch_pos=None, batch_size=2):
        b = batch_size
        self.pixel_pos = pos.unsqueeze(0).repeat(b, 1, 1, 1).to(self.device)
        self.pixel_pos = self.pixel_pos.flatten(2).permute(2, 0, 1)

    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
        return loss_value
    
    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv

    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value
        
    def process(self, data, pdata, half_fmask, ite):
        self.iteration += 1

        self.ite = ite

        mask = 1 - half_fmask
        # zero optimizers
        self.g_optimizer.zero_grad()

        input = torch.cat((pdata, mask), dim=1)
        o = self.g(input, pixel_pos=self.pixel_pos)

        # total
        g_loss = 0

        data_feats = self.lossNet(data)
        o_feats = self.lossNet(o)
        
        # g l1 loss
        g_l1_loss = self.l1_loss(o * half_fmask, data * half_fmask) * self.config.G2_L1_LOSS_WEIGHT
        g_loss += g_l1_loss
        
        g_l1v_loss = self.l1_loss(o * (1 - half_fmask), data * (1 - half_fmask))
        g_loss += g_l1v_loss

        g_col_loss = self.color_loss(o, data) * self.config.G2_COLOR_LOSS_WEIGHT
        g_loss += g_col_loss

        g_perc_loss = self.preceptual_loss(data_feats, o_feats) * self.config.G1_CONTENT_LOSS_WEIGHT
        g_sty_loss = self.style_loss(data_feats, o_feats) * self.config.G2_STYLE_LOSS_WEIGHT
        # g_tv_loss = self.TV_loss(o * half_fmask) * 0.1

        # g_perc_loss, g_sty_loss = self.content_loss(o, data)
        # g_perc_loss = g_perc_loss * self.config.G1_CONTENT_LOSS_WEIGHT
        # g_sty_loss = g_sty_loss * self.config.G2_STYLE_LOSS_WEIGHT
        # c_loss += g_content_loss
        g_loss += g_perc_loss + g_sty_loss
        
        logs = [
            ("l_l1", g_l1_loss.item()),
            ("l_l1v", g_l1v_loss.item()),
            ("l_col", g_col_loss.item()),
            # ("l_tv", g_tv_loss.item()),
            ("l_perc", g_perc_loss.item()),
            ("l_sty", g_sty_loss.item())
        ]
        return o, g_loss, logs
    
    def forward(self, pdata, half_fmask, pos=None, z=None):
        input = torch.cat((pdata, 1 - half_fmask), dim=1)
        # print(input.shape)
        o = self.g(input, pixel_pos=self.pixel_pos)
        return o

    def backward(self, g_loss):
        g_loss.backward()
        self.g_optimizer.step()
