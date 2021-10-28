import torch
import os
import itertools
import torch.nn.functional as F
from .base_model import BaseModel
from util import util
from . import harmony_networks as networks
from . import base_networks as networks_init
from .patchnce import PatchNCELoss
from util import flops


class dhtModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='instance', netG='DHT', dataset_mode='ihd')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_NCE_R', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
            parser.add_argument('--lambda_NCE_I', type=float, default=2.0, help='weight for NCE loss: NCE(G(X), X)')
            
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.postion_embedding = None
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G','G_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['mask', 'harmonized','comp','real','reflectance','illumination']
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G'] 
        self.opt.device = self.device
        self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.opt)
        
        print(self.netG)  

        if self.isTrain:
            util.saveprint(self.opt, 'netG', str(self.netG))  
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            # self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
            # self.criterionNCE = []
            # for nce_layer in self.nce_layers:
            #     self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            
            # self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
            # if opt.nce_r:
            #     self.netF_R = networks.define_F(opt.input_nc, opt.netF, not opt.no_dropout, opt.init_type, opt.init_gain, opt)
            #     util.saveprint(self.opt, 'netF_R', str(self.netF_R))  
            #     self.model_names.append("F_R")
            #     self.loss_names.append("NCE_R")
            #     self.optimizer_F_R = torch.optim.Adam(self.netF_R.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
            #     self.optimizers.append(self.optimizer_F_R)
            # if opt.nce_i:
            #     self.netF_I = networks.define_F(opt.input_nc, opt.netF, not opt.no_dropout, opt.init_type, opt.init_gain, opt)
            #     util.saveprint(self.opt, 'netF_I', str(self.netF_I))  
            #     self.model_names.append("F_I")
            #     self.loss_names.append("NCE_I")
            #     self.optimizer_F_I = torch.optim.Adam(self.netF_I.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
            #     self.optimizers.append(self.optimizer_F_I)
            # if self.opt.gradient_loss:
            #     self.loss_names = ['G','G_L1',"G_R_grident","G_I_L2",'G_I_smooth']
    def set_position(self, pos, patch_pos=None):
        b = self.opt.batch_size
        self.pixel_pos = pos.unsqueeze(0).repeat(b, 1, 1, 1).to(self.device)
        self.pixel_pos = self.pixel_pos.flatten(2).permute(2, 0, 1)
        self.patch_pos = patch_pos.unsqueeze(0).repeat(b, 1, 1, 1).to(self.device)
        self.patch_pos = self.patch_pos.flatten(2).permute(2, 0, 1)
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.inputs = input['inputs'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.image_paths = input['img_path']
        self.mask_r = F.interpolate(self.mask, size=[64,64])
        self.revert_mask = 1-self.mask
        # self.real_mask = torch.zeros_like(self.mask).to(self.device)
        # self.real_input = torch.cat([self.real, self.real_mask], dim=1)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        pass


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        # flops.print_model_flops(self.netG, inputs=self.inputs, image=self.comp*self.revert_mask, pixel_pos=self.pixel_pos.detach(), patch_pos=self.patch_pos.detach(), mask_r=self.mask_r, mask=self.mask)
        # flops.print_model_params(net
        self.harmonized, self.reflectance, self.illumination = self.netG(inputs=self.inputs, image=self.comp*self.revert_mask, pixel_pos=self.pixel_pos.detach(), patch_pos=self.patch_pos.detach(), mask_r=self.mask_r, mask=self.mask)
        if not self.isTrain:
            self.harmonized = self.comp*(1-self.mask) + self.harmonized*self.mask

    def compute_G_loss(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_L1 = self.criterionL1(self.harmonized, self.real)*self.opt.lambda_L1
        self.loss_G = self.loss_G_L1
        # if self.opt.nce_r:
        #     self.reflectance_in = torch.cat([self.reflectance, self.real_mask], dim=1)
        #     self.loss_NCE_R = self.calculate_NCE_loss(self.real_input, self.reflectance_in, self.netF_R, distance_type=self.opt.nce_r_distance_type)
        #     self.loss_G = self.loss_G+self.loss_NCE_R*self.opt.lambda_NCE_R

        # if self.opt.nce_i:
        #     self.illumination_in = torch.cat([self.illumination, self.real_mask], dim=1)
        #     self.loss_NCE_I = self.calculate_NCE_loss(self.real_input, self.illumination_in, self.netF_I, distance_type=self.opt.nce_i_distance_type)
        #     self.loss_G = self.loss_G+self.loss_NCE_I*self.opt.lambda_NCE_I

        # if self.opt.gradient_loss:
        #     self.loss_G_R_grident = self.gradient_loss(self.reflectance, self.real)*10.0
        #     self.loss_G_I_L2 = self.criterionL2(self.illumination, self.real)*10.0
        #     self.loss_G_I_smooth = util.compute_smooth_loss(self.illumination)*1.0
        #     self.loss_G = self.loss_G_L1 + self.loss_G_R_grident + self.loss_G_I_L2 + self.loss_G_I_smooth

        return self.loss_G

    def calculate_NCE_loss(self, src, tgt, net, distance_type='dot'):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(inputs=tgt, layers=self.nce_layers, encode_only=True)

        # if self.opt.flip_equivariance and self.flipped_for_equivariance:
        #     feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(inputs=src, layers=self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = net(feat_k, self.opt.num_patches, None, fg_mask=self.mask)
        feat_q_pool, _ = net(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k, distance_type=distance_type)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers
 
    def optimize_parameters(self):
        # forward
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        # if self.opt.nce_r:
        #     self.optimizer_F_R.zero_grad()
        # if self.opt.nce_i:
        #     self.optimizer_F_I.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        # if self.opt.nce_r:
        #     self.optimizer_F_R.step()
        # if self.opt.nce_i:
        #     self.optimizer_F_I.step()

    def gradient_loss(self, input_1, input_2):
        g_x = self.criterionL1(util.gradient(input_1, 'x'), util.gradient(input_2, 'x'))
        g_y = self.criterionL1(util.gradient(input_1, 'y'), util.gradient(input_2, 'y'))
        return g_x+g_y
    
