import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.optim import lr_scheduler
from torchvision import models
from util.tools import *
from util import util
from . import base_networks as networks_init
from . import transformer
import math

def define_G(netG='retinex',init_type='normal', init_gain=0.02, opt=None):
    """Create a generator
    """
    if netG == 'CNNHT':
        net = CNNHTGenerator(opt)
    elif netG == 'HT':
        net = HTGenerator(opt)
    elif netG == 'DHT':
        net = DHTGenerator(opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    net = networks_init.init_weights(net, init_type, init_gain)
    net = networks_init.build_model(opt, net)
    return net


class HTGenerator(nn.Module):
    def __init__(self, opt=None):
        super(HTGenerator, self).__init__()
        dim = 256
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = opt.ksize, p2 = opt.ksize),
            nn.Linear(opt.ksize*opt.ksize*(opt.input_nc+1), dim)
        )
        self.transformer_enc = transformer.TransformerEncoders(dim, nhead=opt.tr_r_enc_head, num_encoder_layers=opt.tr_r_enc_layers, dim_feedforward=dim*opt.dim_forward, activation=opt.tr_act)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, opt.output_nc, kernel_size=opt.ksize, stride=opt.ksize, padding=0),
            nn.Tanh()
        )
    def forward(self, inputs, pixel_pos=None):
        patch_embedding = self.patch_to_embedding(inputs)
        content = self.transformer_enc(patch_embedding.permute(1, 0, 2), src_pos=pixel_pos)
        bs, L, C  = patch_embedding.size()
        harmonized = self.dec(content.permute(1,2,0).view(bs, C, int(math.sqrt(L)), int(math.sqrt(L))))
        return harmonized

class CNNHTGenerator(nn.Module):
    def __init__(self, opt=None):
        super(CNNHTGenerator, self).__init__()
        dim = 256
        self.enc = ContentEncoder(opt.n_downsample, 0, opt.input_nc+1, dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.transformer_enc = transformer.TransformerEncoders(dim, nhead=opt.tr_r_enc_head, num_encoder_layers=opt.tr_r_enc_layers, dim_feedforward=dim*opt.dim_forward, activation=opt.tr_act)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, opt.output_nc, kernel_size=opt.ksize, stride=opt.ksize, padding=0),
            nn.Tanh()
        )
        if opt.use_two_dec:
            self.dec = ContentDecoder(opt.n_downsample, 0, dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

    def forward(self, inputs, pixel_pos=None):
        content = self.enc(inputs)
        bs,c,h,w = content.size()
        content = self.transformer_enc(content.flatten(2).permute(2, 0, 1), src_pos=pixel_pos)
        harmonized = self.dec(content.permute(1, 2, 0).view(bs, c, h, w))
        return harmonized

class DHTGenerator(nn.Module):
    def __init__(self, opt=None):
        super(DHTGenerator, self).__init__()
        self.reflectance_dim = 256
        self.device = opt.device
        self.reflectance_enc = ContentEncoder(opt.n_downsample, 0, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.reflectance_dec = ContentDecoder(opt.n_downsample, 0, self.reflectance_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

        self.reflectance_transformer_enc = transformer.TransformerEncoders(self.reflectance_dim, nhead=opt.tr_r_enc_head, num_encoder_layers=opt.tr_r_enc_layers, dim_feedforward=self.reflectance_dim*opt.dim_forward, activation=opt.tr_act)

        self.light_generator = GlobalLighting(light_element=opt.light_element,light_mlp_dim=self.reflectance_dim, opt=opt)
        self.illumination_render= transformer.TransformerDecoders(self.reflectance_dim, nhead=opt.tr_i_dec_head, num_decoder_layers=opt.tr_i_dec_layers, dim_feedforward=self.reflectance_dim*opt.dim_forward, activation=opt.tr_act)
        self.illumination_dec = ContentDecoder(opt.n_downsample, 0, self.reflectance_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)
        self.opt = opt
    def forward(self, inputs=None, image=None, pixel_pos=None, patch_pos=None, mask_r=None, mask=None, layers=[], encode_only=False):
        r_content = self.reflectance_enc(inputs)
        bs,c,h,w = r_content.size()

        reflectance = self.reflectance_transformer_enc(r_content.flatten(2).permute(2, 0, 1), src_pos=pixel_pos, src_key_padding_mask=None)
        
        light_code, light_embed = self.light_generator(image, pos=patch_pos, mask=mask, use_mask=self.opt.light_use_mask)
        illumination = self.illumination_render(light_code, reflectance, src_pos=light_embed, tgt_pos=pixel_pos, src_key_padding_mask=None, tgt_key_padding_mask=None)

        reflectance = reflectance.permute(1, 2, 0).view(bs, c, h, w)
        reflectance = self.reflectance_dec(reflectance)
        reflectance = reflectance / 2 +0.5

        illumination = illumination.permute(1, 2, 0).view(bs, c, h, w)
        illumination = self.illumination_dec(illumination)
        illumination = illumination / 2 + 0.5
        
        harmonized = reflectance*illumination
        return harmonized, reflectance, illumination


class GlobalLighting(nn.Module):
    def __init__(self, light_element=27, light_mlp_dim=8, norm=None, activ=None, pad_type='zero', opt=None):
    
        super(GlobalLighting, self).__init__()
        self.light_with_tre = opt.light_with_tre

        patch_size = opt.patch_size
        image_size = opt.crop_size
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = opt.input_nc * patch_size ** 2
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        )
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, light_mlp_dim),
        )
        dim = light_mlp_dim
        if opt.light_with_tre:
            self.transformer_enc = transformer.TransformerEncoders(dim, nhead=opt.tr_l_enc_head, num_encoder_layers=opt.tr_l_enc_layers, dim_feedforward=dim*2, dropout=0.0, activation=opt.tr_act)
        self.transformer_dec = transformer.TransformerDecoders(dim, nhead=opt.tr_l_dec_head, num_decoder_layers=opt.tr_l_dec_layers, dim_feedforward=dim*2, dropout=0.0, activation=opt.tr_act)
        self.light_embed = nn.Embedding(light_element, dim)

    def forward(self, inputs, mask=None, pos=None, multiple=False, use_mask=False):
        b,c,h,w = inputs.size()
        light_embed = self.light_embed.weight.unsqueeze(1).repeat(1, b, 1)
        tgt = torch.zeros_like(light_embed)
        input_patch = self.patch_embedding(self.to_patch(inputs))
        input_patch = input_patch.permute(1,0,2)
        src_key_padding_mask = None
        if use_mask:
            mask_patch = self.to_patch(mask)
            mask_sum = torch.sum(mask_patch, dim=2)
            src_key_padding_mask = mask_sum.to(bool)
        if self.light_with_tre:
            input_patch = self.transformer_enc(input_patch, src_pos=pos, src_key_padding_mask=src_key_padding_mask)
        light = self.transformer_dec(input_patch, tgt, src_pos=pos, tgt_pos=light_embed, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=None)        
        return light, light_embed

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
           
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm='ln', activation=activ, pad_type=pad_type)]
        if not dim == output_dim:
            self.model += [Conv2dBlock(dim, output_dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

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
        return self.model(x)

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
##################################################################################
# Basic Blocks
##################################################################################
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

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(ConvTranspose2dBlock, self).__init__()
        self.use_bias = True
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
        elif norm == 'adain_dyna':
            self.norm = AdaptiveInstanceNorm2d_Dyna(norm_dim)
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
            self.conv = SpectralNorm(nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias))
        else:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
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

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


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
