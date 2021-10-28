import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
import numpy as np
from torch.optim import lr_scheduler
from torch.nn.utils import spectral_norm
from util import util
from . import transformer

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x
def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer
class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1, last_epoch=opt.epoch_count)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return net

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], opt=None):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    # net = build_model(opt, net, gpu_id=gpu_ids[0])
    init_weights(net, init_type, init_gain=init_gain)
    # net = build_model(opt, net, gpu_id=gpu_ids[0])
    return net

def build_model(cfg, model, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    # name = cfg.MODEL.MODEL_NAME
    # model = MODEL_REGISTRY.get(name)(cfg)
    
    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        #,  find_unused_parameters=True
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    
    return model



def define_D(netD='n_layers',init_type='normal', init_gain=0.02, opt=None):

    norm_layer = get_norm_layer(norm_type='instance')
    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(opt.input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(opt.input_nc, opt.ndf, n_layers=opt.n_layers_D, norm_layer=norm_layer)
    elif netD == 'patch_tf':  # more options
        net = NLayerTFDiscriminator(opt.input_nc, opt.ndf, n_layers=opt.n_layers_D, norm_layer=norm_layer)
    elif netD == 'fcn':
        net = FCNDiscriminator(opt.input_nc, opt.ndf, n_layers=5, norm_layer=norm_layer)
    elif netD == 'conv':
        net = ConvDiscriminator(opt.output_nc, opt.ndf, opt.n_layers_D, norm_layer)
    elif netD == 'patchco':
        net = PatchCoDiscriminator(opt.output_nc, opt.ndf, opt.n_layers_D, norm_layer)
    elif netD == 'glconv':
        net = GLConvDiscriminator(opt.output_nc, opt.ndf, opt.n_layers_D, norm_layer)
    # return init_net(net, init_type, init_gain, gpu_ids)
    net = init_weights(net, init_type, init_gain)
    net = build_model(opt, net)
    return net

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'l1':
            self.loss = nn.L1Loss()
        elif gan_mode in ['wgangp']:
            self.loss = None
            self.relu = nn.ReLU()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla','l1']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean() # self.relu(1-prediction.mean())
            else:
                loss = prediction.mean() # self.relu(1+prediction.mean())
       
        return loss


class SNPatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self,opt):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(SNPatchDiscriminator, self).__init__()
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        n_layers = 3
        ndf = opt.ndf
        use_bias = True
        sequence = [nn.utils.spectral_norm(nn.Conv2d(opt.input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                # norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        output = self.model(input)
        output = torch.flatten(output, start_dim=1)
        return output


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class OrgDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d, global_stages=0):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(OrgDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 0
        self.conv1 = spectral_norm(PartialConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
        if global_stages < 1:
            self.conv1f = spectral_norm(PartialConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
        else:
            self.conv1f = self.conv1
        self.relu1 = nn.LeakyReLU(0.2, True)
        nf_mult = 1
        nf_mult_prev = 1

        n = 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv2 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm2 = norm_layer(ndf * nf_mult)
        if global_stages < 2:
            self.conv2f = spectral_norm(
                PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            self.norm2f = norm_layer(ndf * nf_mult)
        else:
            self.conv2f = self.conv2
            self.norm2f = self.norm2

        self.relu2 = nn.LeakyReLU(0.2, True)

        n = 2
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv3 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm3 = norm_layer(ndf * nf_mult)
        if global_stages < 3:
            self.conv3f = spectral_norm(
                PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            self.norm3f = norm_layer(ndf * nf_mult)
        else:
            self.conv3f = self.conv3
            self.norm3f = self.norm3
        self.relu3 = nn.LeakyReLU(0.2, True)

        n = 3
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.norm4 = norm_layer(ndf * nf_mult)
        self.conv4 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.conv4f = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm4f = norm_layer(ndf * nf_mult)

        self.relu4 = nn.LeakyReLU(0.2, True)

        n = 4
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv5 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.conv5f = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm5 = norm_layer(ndf * nf_mult)
        self.norm5f = norm_layer(ndf * nf_mult)
        self.relu5 = nn.LeakyReLU(0.2, True)

        n = 5
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv6 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.conv6f = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm6 = norm_layer(ndf * nf_mult)
        self.norm6f = norm_layer(ndf * nf_mult)
        self.relu6 = nn.LeakyReLU(0.2, True)

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.conv7 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        self.conv7f = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))

    def forward(self, input, mask=None):
        x = input
        x, _ = self.conv1(x)
        x = self.relu1(x)
        x, _ = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x, _ = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x, _ = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        x, _ = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)
        x, _ = self.conv6(x)
        x = self.norm6(x)
        x = self.relu6(x)
        x, _ = self.conv7(x)

        """Standard forward."""
        xf, xb = input, input
        mf, mb = mask, 1 - mask

        xf, mf = self.conv1f(xf, mf)
        xf = self.relu1(xf)
        xf, mf = self.conv2f(xf, mf)
        xf = self.norm2f(xf)
        xf = self.relu2(xf)
        xf, mf = self.conv3f(xf, mf)
        xf = self.norm3f(xf)
        xf = self.relu3(xf)
        xf, mf = self.conv4f(xf, mf)
        xf = self.norm4f(xf)
        xf = self.relu4(xf)
        xf, mf = self.conv5f(xf, mf)
        xf = self.norm5f(xf)
        xf = self.relu5(xf)
        xf, mf = self.conv6f(xf, mf)
        xf = self.norm6f(xf)
        xf = self.relu6(xf)
        xf, mf = self.conv7f(xf, mf)

        xb, mb = self.conv1f(xb, mb)
        xb = self.relu1(xb)
        xb, mb = self.conv2f(xb, mb)
        xb = self.norm2f(xb)
        xb = self.relu2(xb)
        xb, mb = self.conv3f(xb, mb)
        xb = self.norm3f(xb)
        xb = self.relu3(xb)
        xb, mb = self.conv4f(xb, mb)
        xb = self.norm4f(xb)
        xb = self.relu4(xb)
        xb, mb = self.conv5f(xb, mb)
        xb = self.norm5f(xb)
        xb = self.relu5(xb)
        xb, mb = self.conv6f(xb, mb)
        xb = self.norm6f(xb)
        xb = self.relu6(xb)
        xb, mb = self.conv7f(xb, mb)

        return x, xf, xb


class DovenetNLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(DovenetNLayerDiscriminator, self).__init__()
        num_outputs = ndf * min(2 ** n_layers, 8)
        self.D = OrgDiscriminator(input_nc, ndf, n_layers, norm_layer)
        self.convl1 = spectral_norm(nn.Conv2d(num_outputs, num_outputs, kernel_size=1, stride=1))
        self.relul1 = nn.LeakyReLU(0.2)
        self.convl2 = spectral_norm(nn.Conv2d(num_outputs, num_outputs, kernel_size=1, stride=1))
        self.relul2 = nn.LeakyReLU(0.2)
        self.convl3 = nn.Conv2d(num_outputs, 1, kernel_size=1, stride=1)
        self.convg3 = nn.Conv2d(num_outputs, 1, kernel_size=1, stride=1)

    def forward(self, input, mask=None, gp=False, feat_loss=False):

        x, xf, xb = self.D(input, mask)
        feat_l, feat_g = torch.cat([xf, xb]), x
        x = self.convg3(x)

        sim = xf * xb
        sim = self.convl1(sim)
        sim = self.relul1(sim)
        sim = self.convl2(sim)
        sim = self.relul2(sim)
        sim = self.convl3(sim)
        sim_sum = sim
        if not gp:
            if feat_loss:
                return x, sim_sum, feat_g, feat_l
            return x, sim_sum
        return  (x + sim_sum) * 0.5

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0, mask=None):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv, mask, gp=True)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True,
                                        allow_unused=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=True):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class NLayerTFDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=True):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerTFDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.encoder = nn.Sequential(*sequence)
        dim = ndf * nf_mult
        self.transformer_enc = transformer.TransformerDecoders(dim, nhead=4, num_encoder_layers=4, dim_feedforward=dim*2, dropout=0.0)

        self.query_embed = nn.Embedding(1, dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.LayerNorm(dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, dim//4),
            nn.LayerNorm(dim//4),
            nn.ReLU(),
            nn.Linear(dim//4, 1),
            nn.Sigmoid()
        )
    def forward(self, input, pos, mask_r):
        """Standard forward."""
        output = self.encoder(input)
        bs, c, h, w = output.size()
        fg_key_padding_mask = mask_r.flatten(1).to(torch.bool)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)+0.5
        tf_bg_output = self.transformer_enc(output.flatten(2).permute(2, 0, 1), tgt, src_key_padding_mask=fg_key_padding_mask, src_pos=pos, tgt_pos=query_embed)
        tf_all_output = self.transformer_enc(output.flatten(2).permute(2, 0, 1), tgt, src_key_padding_mask=None, src_pos=pos, tgt_pos=query_embed)

        return self.classifier(tf_bg_output.view(bs, -1)), self.classifier(tf_all_output.view(bs, -1))




class FCNDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(FCNDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        # sequence += [
        #     nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
        #     norm_layer(ndf * nf_mult),
        #     nn.LeakyReLU(0.2, True)
        # ]
        self.model = nn.Sequential(*sequence)
        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.fcn = nn.Sequential(nn.ConvTranspose2d(ndf*nf_mult, 1, kernel_size=32, stride=32, padding=0),
                                nn.Sigmoid()
                                )
        self.downConv = nn.Sequential(
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=4, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=padw)
        )
        

    def forward(self, input):
        """Standard forward."""
        down_output = self.model(input)
        fcn_output = self.fcn(down_output)
        global_output = self.downConv(down_output)
        return fcn_output, global_output
        # return self.model(input)

class ConvDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(ConvDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # nf_mult_prev = nf_mult
        # nf_mult = min(2 ** n_layers, 8)
        # sequence += [
        #     nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
        #     norm_layer(ndf * nf_mult),
        #     nn.LeakyReLU(0.2, True)
        # ]
        self.model = nn.Sequential(*sequence)
        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.fcn = nn.Sequential(
            nn.ConvTranspose2d(ndf*nf_mult, ndf*nf_mult//2, kernel_size=32, stride=32, padding=0),
            norm_layer(ndf * nf_mult//2),
            nn.LeakyReLU(0.2, True)
        )
        self.linear = nn.Sequential(
            nn.Linear(ndf*nf_mult//2, 8)
        )
        

    def forward(self, input, fg_mask):
        """Standard forward."""
        down_output = self.model(input)
        fcn_output = self.fcn(down_output)

        N, C, H, W = fcn_output.size()
        fg_mask_sum = torch.sum(fg_mask.view(N, 1, -1), dim=2)
        bg_mask_sum = H*W - fg_mask_sum
        fg_mask_sum = fg_mask_sum
        bg = fcn_output*(1-fg_mask)
        fg = fcn_output*fg_mask

        # avg pooling to 1*1
        bg_pooling = torch.sum(bg.view(N,C,-1), dim=2).div(bg_mask_sum)
        fg_pooling = torch.sum(fg.view(N,C,-1), dim=2).div(fg_mask_sum)

        bg_distribution = self.linear(bg_pooling)
        fg_distribution = self.linear(fg_pooling)
        bg_distribution = bg_distribution.unsqueeze(-1)
        fg_distribution = fg_distribution.unsqueeze(-1)

        bg_distribution_mu = torch.mean(bg_distribution, dim=1, keepdim=True)
        fg_distribution_mu = torch.mean(fg_distribution, dim=1, keepdim=True)

        # bg_distribution_var = torch.var(bg_distribution, dim=1, keepdim=True) + 1e-8
        # fg_distribution_var = torch.var(fg_distribution, dim=1, keepdim=True) + 1e-8

        bg_distribution_std = (torch.var(bg_distribution, dim=1, keepdim=True) + 1e-8).sqrt()
        fg_distribution_std = (torch.var(fg_distribution, dim=1, keepdim=True) + 1e-8).sqrt()

        fg_bg_conv = torch.matmul((bg_distribution-bg_distribution_mu).permute(0,2,1), (fg_distribution-fg_distribution_mu))/C
        fg_bg_r = fg_bg_conv.div(bg_distribution_std*fg_distribution_std+1e-8)
        fg_bg_r = fg_bg_r.squeeze(-1)
        return fg_bg_r

class LLConvDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(LLConvDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=7, stride=1, padding=3),
            norm_layer(ndf), 
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # nf_mult_prev = nf_mult
        # nf_mult = min(2 ** n_layers, 8)
        # sequence += [
        #     nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
        #     norm_layer(ndf * nf_mult),
        #     nn.LeakyReLU(0.2, True)
        # ]
        sequence += [nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=3, stride=1, padding=1)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        

    def forward(self, input, fg_mask):
        """Standard forward."""
        down_output = self.model(input)

        N, C, H, W = down_output.size()
        fg_mask_sum = torch.sum(fg_mask.view(N, 1, -1), dim=2, keepdim=True)
        bg_mask_sum = H*W - fg_mask_sum+1e-8
        fg_mask_sum = fg_mask_sum+1e-8

        bg = down_output*(1-fg_mask)
        fg = down_output*fg_mask

        # avg pooling to 1*1
        bg_pooling = torch.sum(bg.view(N,C,-1), dim=2, keepdim=True).div(bg_mask_sum) #each channel mean  == channel avg pooling
        fg_pooling = torch.sum(fg.view(N,C,-1), dim=2, keepdim=True).div(fg_mask_sum)

        bg_distribution_mu = torch.mean(bg_pooling, dim=1, keepdim=True)  #all channel mean
        fg_distribution_mu = torch.mean(fg_pooling, dim=1, keepdim=True)  #all channel mean

        bg_distribution_std = (torch.var(bg_pooling, dim=1, keepdim=True) + 1e-8).sqrt()
        fg_distribution_std = (torch.var(fg_pooling, dim=1, keepdim=True) + 1e-8).sqrt()

        fg_bg_conv = torch.matmul((bg_pooling-bg_distribution_mu).permute(0,2,1), (fg_pooling-fg_distribution_mu))/C
        fg_bg_r = fg_bg_conv.div(bg_distribution_std*fg_distribution_std+1e-8)
        fg_bg_r = fg_bg_r.squeeze(-1)
        fg_bg_r = fg_bg_r.abs()
        return fg_bg_r
class GLConvDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(GLConvDiscriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [
            DiscriminatorBlock(input_nc, ndf, downsample=True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                DiscriminatorBlock(ndf * nf_mult_prev, ndf * nf_mult)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        # sequence += [DiscriminatorBlock(ndf * nf_mult_prev, ndf * nf_mult, downsample=False)]
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(ndf*nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.model = nn.Sequential(*sequence)
        self.patchgan_conv = nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=1)
        # self.global_conv = nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs, fg_mask=None):
        """Standard forward."""
        features = self.model(inputs)
        patch_gan_output = self.patchgan_conv(features)
        return patch_gan_output, None
        # global_output = self.global_conv(features)
        
        # return patch_gan_output, global_output

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1)),
            nn.InstanceNorm2d(filters),
        )
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            nn.InstanceNorm2d(filters),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(filters, filters, 3, padding=1, stride = 2),
            nn.InstanceNorm2d(filters),
            nn.LeakyReLU(0.2, True)
            
        )

        # self.downsample = nn.Sequential(
        #     Blur(),
        #     nn.Conv2d(filters, filters, 3, padding = 1, stride = 2)
        #     # nn.InstanceNorm2d(filters),
        #     nn.LeakyReLU(0.2, True),

        # ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        # if exists(self.downsample):
        #     x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x
class DiscriminatorResBlock(nn.Module):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(input_channels, filters, 1, stride = 1),
            nn.InstanceNorm2d(filters),
        )
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            nn.InstanceNorm2d(filters),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(filters, filters, 3, padding=1, stride = 1),
            nn.InstanceNorm2d(filters),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class PatchCoDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchCoDiscriminator, self).__init__()
        

        self.ksizes = 8
        self.strides = 8
        self.paddingsize = 0
        dim = self.ksizes*self.ksizes*input_nc
        self.transformer_enc = transformer.TransformerEncoders(dim, nhead=4, num_encoder_layers=4, dim_feedforward=dim*2, dropout=0.1)
        self.transformer_dec = transformer.TransformerDecoders(dim, nhead=1, num_encoder_layers=4, dim_feedforward=dim*2, dropout=0.1)
        # self.output_layers = nn.Sequential(
        #     # nn.Linear(dim, dim//2),
        #     # nn.LayerNorm(dim//2),
        #     # nn.ReLU(),
        #     nn.Linear(dim, 4)
        # )
        self.query_embed = nn.Embedding(3, dim)
    def forward(self, inputs, fg_mask=None, postion_embedding=None):
        """Standard forward."""
        b,c,h,w = inputs.size()
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)
        # postion_embedding = self.pos_embed.unsqueeze(0).repeat(b, 1, 1, 1).flatten(2).permute(2, 0, 1).to(inputs.device)

        tgt = torch.zeros_like(query_embed)
        # fg_mask = 1-fg_mask
        input_mask_patch = util.extract_image_patches(torch.cat((inputs,fg_mask),dim=1), ksizes=[self.ksizes, self.ksizes],
                                strides=[self.strides, self.strides],
                                rates=[1, 1],
                                padding='valid',
                                paddingsize=self.paddingsize)     #1 2304 1024  [N, C*k*k, L]
        L = input_mask_patch.size(2)
        input_mask_patch = input_mask_patch.permute(0,2,1).view(b, L, c+1, self.ksizes*self.ksizes)
        input_patch = input_mask_patch[:,:,:3,:]
        mask_patch = input_mask_patch[:,:,3:,:].squeeze(2)
        # input_patch, mask_patch = input_mask_patch[0], input_mask_patch[1].squeeze(2)
        # mask_patch_sum = torch.sum(mask_patch, dim=2)
        # fg_mask_patch_pos = torch.nonzero(mask_patch_sum)
        
        # patch_mask = torch.zeros([b, L], device=inputs.device)
        # patch_mask[:,fg_mask_patch_pos[:]] = 1
        # fg_patch_mask = patch_mask
        mask_patch_sum = torch.sum(mask_patch, dim=2).flatten(0)
        fg_mask_patch_pos = torch.nonzero(mask_patch_sum)
        
        patch_mask = torch.zeros(b*L, device=inputs.device)
        patch_mask[fg_mask_patch_pos[:]] = 1
        fg_patch_mask = patch_mask.view(b,L)

        mask_pixel_fg = mask_patch.unsqueeze(2).repeat(1,1,3,1)
        # transformer_input = input_patch.flatten(2).permute(1, 0, 2)
        bg_patch_mask = 1 - fg_patch_mask
        bg_transformer_input = (input_patch*(1-mask_pixel_fg)).flatten(2).permute(1, 0, 2)
        bg_enc = self.transformer_enc(bg_transformer_input, src_pos=postion_embedding, src_key_padding_mask=fg_patch_mask.to(torch.bool))
        bg_dec = self.transformer_dec(bg_enc, tgt, src_pos=postion_embedding, tgt_pos=query_embed, src_key_padding_mask=fg_patch_mask.to(torch.bool), tgt_key_padding_mask=None)
        # bg_dec = self.output_layers(bg_dec.permute(1,0,2))
        bg_dec = bg_dec.permute(1,0,2)
        fg_transformer_input = (input_patch*mask_pixel_fg).flatten(2).permute(1, 0, 2)
        fg_enc = self.transformer_enc(fg_transformer_input, src_pos=postion_embedding, src_key_padding_mask=bg_patch_mask.to(torch.bool))
        fg_dec = self.transformer_dec(fg_enc, tgt, src_pos=postion_embedding, tgt_pos=query_embed, src_key_padding_mask=bg_patch_mask.to(torch.bool), tgt_key_padding_mask=None)
        # fg_dec = self.output_layers(fg_dec.permute(1,0,2))
        fg_dec = fg_dec.permute(1,0,2)

        bg_color = F.softmax(bg_dec[:,:3,:], dim=1)
        fg_color = F.softmax(fg_dec[:,:3,:], dim=1)
        loss = self.same_loss(bg_color, fg_color)*0.3 + self.same_loss(bg_dec.mean(dim=1),fg_dec.mean(dim=1))*0.7
        # bg_color = F.softmax(bg_dec[:,:3], dim=1)
        # fg_color = F.softmax(fg_dec[:,:3], dim=1)
        # loss = self.same_loss(bg_color, fg_color) + self.same_loss(bg_dec[:,3:],fg_dec[:,3:])
        return loss

        # return self.sigmoid(torch.matmul(bg_dec, fg_dec))
        bg_distribution_mu = torch.mean(bg_dec, dim=2, keepdim=True)
        fg_distribution_mu = torch.mean(fg_dec, dim=2, keepdim=True)
        bg_distribution_std = (torch.var(bg_dec, dim=2, keepdim=True) + 1e-8).sqrt()
        fg_distribution_std = (torch.var(fg_dec, dim=2, keepdim=True) + 1e-8).sqrt()
        fg_bg_conv = torch.matmul((bg_dec-bg_distribution_mu), (fg_dec-fg_distribution_mu).permute(0,2,1))/768
        fg_bg_r = fg_bg_conv.div(bg_distribution_std*fg_distribution_std+1e-8)
        fg_bg_r = fg_bg_r.squeeze(-1)
        fg_bg_r = fg_bg_r.abs()
        # fg_bg_r = torch.tanh(fg_bg_r)
        return fg_bg_r
        

        # # avg pooling to 1*1
        # bg_pooling = torch.sum(bg.view(N,C,-1), dim=2, keepdim=True).div(bg_mask_sum) #each channel mean  == channel avg pooling
        # fg_pooling = torch.sum(fg.view(N,C,-1), dim=2, keepdim=True).div(fg_mask_sum)

        # bg_distribution_mu = torch.mean(bg_pooling, dim=1, keepdim=True)  #all channel mean
        # fg_distribution_mu = torch.mean(fg_pooling, dim=1, keepdim=True)  #all channel mean

        # bg_distribution_std = (torch.var(bg_pooling, dim=1, keepdim=True) + 1e-8).sqrt()
        # fg_distribution_std = (torch.var(fg_pooling, dim=1, keepdim=True) + 1e-8).sqrt()

        # fg_bg_conv = torch.matmul((bg_pooling-bg_distribution_mu).permute(0,2,1), (fg_pooling-fg_distribution_mu))/C
        # fg_bg_r = fg_bg_conv.div(bg_distribution_std*fg_distribution_std+1e-8)
        # fg_bg_r = fg_bg_r.squeeze(-1)
        # fg_bg_r = fg_bg_r.abs()
        # return fg_bg_r


        return output
    def same_loss(self, x, y):
        C1 = 0.01**2
        loss = 2*x.mul(y).div(x.pow(2)+y.pow(2)+C1).flatten(1)
        loss = torch.tanh(loss.pow(2).mean(dim=1))
        return loss

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2D(x, f, normalized=True)

class GANLocalLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLocalLoss, self).__init__()
        # self.pooling = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.adaptivepooling = nn.AdaptiveAvgPool2d(64)

    def __call__(self, prediction, fg_mask, image):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        # prediction = self.pooling(prediction)
        # fg_mask = self.pooling(fg_mask)
        N, C, H, W = prediction.size()
        bg = prediction*(1-fg_mask)
        fg = prediction*fg_mask


        fg_patch = fg.view(N,C,-1).permute(0,2,1)
        bg_patch = bg.view(N,C,-1)

        fg_patch_mu = torch.mean(fg_patch, dim=2, keepdim=True)
        bg_patch_mu = torch.mean(bg_patch, dim=1, keepdim=True)
        fg_bg_local_conv = torch.matmul((fg_patch-fg_patch_mu), (bg_patch-bg_patch_mu))/(C-1)

        bg_distribution_std = (torch.var(bg_patch, dim=1, keepdim=True) + 1e-8).sqrt()
        fg_distribution_std = (torch.var(fg_patch, dim=2, keepdim=True) + 1e-8).sqrt()
        fg_bg_r = fg_bg_local_conv.div(torch.matmul(fg_distribution_std,bg_distribution_std)+1e-8)
        fg_bg_r = fg_bg_r.abs()
        # fg_bg_r[fg_bg_r<0.7] = 0

        pixel_count = H*W
        # # bg_patch_one = bg.unsqueeze(1).repeat(1, pixel_count, 1, 1, 1)
        # # fg_patch_one = fg.view(N,C,-1).permute(0,2,1).unsqueeze(-1).unsqueeze(-1).expand_as(bg_patch_one)
        # bg_patch_one = bg.permute(0,2,1,3).permute(0,1,3,2).unsqueeze(1)
        # fg_patch_one = fg.view(N,C,-1).permute(0,2,1).unsqueeze(-2).unsqueeze(-2)
        # fg_bg_L1 = (fg_patch_one-bg_patch_one).pow(2).mean(dim=-1)
        # fg_bg_L1_drop_fg = fg_bg_L1*(1-fg_mask)

        # fg_mask_channel = fg_mask.view(N, -1, 1, 1).expand_as(fg_bg_L1)
        # fg_bg_L1_only_fg = fg_bg_L1_drop_fg*fg_mask_channel

        # # fg_bg_local_conv[fg_bg_local_conv<0] = 0
        # # fg_bg_local_conv = torch.softmax(fg_bg_local_conv, dim=2)
        # # local_loss = fg_bg_L1_only_fg.view(N, pixel_count, pixel_count)*fg_bg_local_conv.permute(0,2,1).detach()
        # local_loss = fg_bg_L1_only_fg.view(N, pixel_count, -1)*fg_bg_r
        # fg_mask_sum = fg_mask.view(N, -1).sum(dim=1)

        C1 = 0.01**2
        image = self.adaptivepooling(image)
        # image = F.adaptive_avg_pool2d(image, 32)
        # print(image.size())
        image_fg = image*fg_mask
        image_bg = image*(1-fg_mask)
        image_fg_mu = image_fg.mean(dim=1)
        image_bg_mu = image_bg.mean(dim=1)
        image_fg_patch_one = image_fg_mu.view(N, -1,1)
        image_bg_patch_one = image_bg_mu.view(N, -1,1)
        image_fg_patch_one_sq = image_fg_patch_one.pow(2)
        image_bg_patch_one_sq = image_bg_patch_one.pow(2)

        luminance = torch.matmul(image_fg_patch_one, image_bg_patch_one.permute(0,2,1)+C1).div(image_fg_patch_one_sq+image_bg_patch_one_sq+C1)
        # image_bg_patch_one = image_bg.permute(0,2,1,3).permute(0,1,3,2).unsqueeze(1)
        # image_fg_patch_one = image_fg.view(N,image_fg.size(1),-1).permute(0,2,1).unsqueeze(-2).unsqueeze(-2)
        # fg_bg_L1 = (image_fg_patch_one-image_bg_patch_one).pow(2).mean(dim=-1)
        fg_bg_loss = luminance
        
        fg_bg_loss_drop_fg = fg_bg_loss*(1-fg_mask.view(N,1, -1))
        fg_mask_channel = fg_mask.view(N, -1, 1).expand_as(fg_bg_loss)
        fg_bg_loss_only_fg = fg_bg_loss_drop_fg*fg_mask_channel
        local_loss = fg_bg_loss_only_fg*fg_bg_r.detach()

        local_loss = local_loss.mean()
        loss = local_loss
        # if target_is_real:
        #     loss = local_loss # self.relu(1-prediction.mean())
        # else:
        #     loss = -local_loss # self.relu(1+prediction.mean())
        return loss
