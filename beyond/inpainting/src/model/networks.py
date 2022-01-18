import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from .partialconv import PartialConv2d
from .reshape import Reshape
import torch.nn.utils.spectral_norm as spectral_norm
from . import transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from .. import utilsgzh as util

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('ConvTranspose2d') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_dim, dim, kernel_size=7, padding=0, bias=False),
                 nn.InstanceNorm2d(dim),
                 nn.LeakyReLU(0.2, True)]
        # downsampling blocks
        for i in range(n_downsample):

            self.model += [nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                          nn.InstanceNorm2d(dim * 2),
                          nn.LeakyReLU(0.2, True),
                        #   Downsample(dim * 2)
                          ]
            dim *= 2
           
        # residual blocks
        # model += [ResBlocks(n_res, dim, norm='ln', activation=activ, pad_type=pad_type)]
        if not dim == output_dim:
            self.model += [Conv2dBlock(dim, output_dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, input, layers=[], encode_only=False):
        # return self.model(x)
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake

class ContentDecoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type):
        super(ContentDecoder, self).__init__()
        self.model = []
        dim = input_dim
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]

        # upsampling blocks
        for i in range(n_downsample):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)
            ]
            dim //= 2

        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return (self.model(x) + 1) / 2
        # return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'adain_ori':
            self.norm = AdaptiveInstanceNorm2d_IN(norm_dim)
        elif norm == 'remove_render':
            self.norm = RemoveRender(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class FCHTGenerator(BaseNetwork):
    def __init__(self, config=None):
        super(FCHTGenerator, self).__init__()
        dim = 256
        ksize = 4
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = ksize, p2 = ksize),
            nn.Linear(ksize*ksize*4, dim)
        )
        self.transformer_enc = transformer.TransformerEncoders(dim, nhead=2, num_encoder_layers=6, dim_feedforward=dim*2, activation='gelu')
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, 3, kernel_size=ksize, stride=ksize, padding=0),
            nn.Tanh()
        )
        
        self.down = nn.UpsamplingNearest2d(scale_factor=.25)
        self.init_weights()
    def forward(self, inputs, pixel_pos=None, mask_r=None):
        if mask_r is not None:
            mask_r = self.down(mask_r).flatten(1).to(torch.bool)
        patch_embedding = self.patch_to_embedding(inputs)
        content = self.transformer_enc(patch_embedding.permute(1, 0, 2), src_pos=pixel_pos, src_key_padding_mask=mask_r)
        bs, L, C  = patch_embedding.size()
        harmonized = self.dec(content.permute(1,2,0).view(bs, C, int(math.sqrt(L)), int(math.sqrt(L))))
        harmonized = (harmonized+1)/2
        return harmonized

