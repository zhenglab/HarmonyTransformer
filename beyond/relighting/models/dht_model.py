import torch
import math
import os
import itertools
import torch.nn.functional as F
from .base_model import BaseModel
from util import util
from . import harmony_networks as networks
from . import base_networks as networks_init
import util.ssim as ssim

class DHTModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='instance', netG='DHT', dataset_mode='dpr')
        parser.add_argument('--ksize', type=int, default=4, help='weight for L1 loss')
        parser.add_argument('--stride', type=int, default=4, help='weight for L1 loss')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_ssim', type=float, default=20., help='weight for L L2 loss')
            
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.postion_embedding = None
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G','G_L1','G_SSIM', "G_L"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['harmonized','fake','real','reflectance','illumination','target']
        
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
            self.criterionSSIM = ssim.SSIM()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
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
        self.image_paths = input['img_path']
        self.fake = input['fake'].to(self.device)
        self.real = input['real'].to(self.device)
        if input['target'] is not None:
            self.target = input['target'].to(self.device)
        if input['fake_light'] is not None:
            self.light_fake = input['fake_light'].to(self.device)
        if input['real_light'] is not None:
            self.light_real = input['real_light'].to(self.device)
        self.image_paths = input['img_path']

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
        if self.isTrain:
            self.harmonized, self.reflectance, self.illumination, self.light_gen_fake = self.netG(inputs=self.fake, pixel_pos=self.pixel_pos.detach(), patch_pos=self.patch_pos.detach(), input_light=self.light_real)
        else:
            if self.opt.relighting_action == "relighting":
                self.harmonized, self.reflectance, self.illumination, self.light_gen_fake = self.netG(self.fake, pixel_pos=self.pixel_pos.detach(), patch_pos=self.patch_pos.detach(), isTest=True, input_light=self.light_real)
            else:
                self.harmonized, self.reflectance, self.illumination, self.light_gen_fake = self.netG(self.fake, pixel_pos=self.pixel_pos.detach(), patch_pos=self.patch_pos.detach(), isTest=True, target=self.target)
    def compute_G_loss(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_L1 = self.criterionL1(self.harmonized, self.real)*self.opt.lambda_L1
        self.loss_G_L = self.criterionL2(self.light_gen_fake, self.light_fake)*self.opt.lambda_L
        self.loss_G_SSIM = (1-self.criterionSSIM(self.harmonized, self.real))*self.opt.lambda_ssim
        self.loss_G = self.loss_G_L1 + self.loss_G_SSIM + self.loss_G_L

        return self.loss_G

    def optimize_parameters(self):
        # forward
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

