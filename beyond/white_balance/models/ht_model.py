import torch
import os
import itertools
import torch.nn.functional as F
from .base_model import BaseModel
from util import util
from util import utils
from . import harmony_networks as networks
from . import base_networks as networks_init


class HTModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='HTTRDFC', dataset_mode='wb')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.postion_embedding = None
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G','AWB_L1','T_L1','S_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['image', 'AWB','gt_AWB','T',"gt_T",'S',"gt_S"]
        if not self.isTrain:
            # self.visual_names = ['image', 'AWB', 'T', 'S','F','D','C']
            self.visual_names = ['image', 'AWB', 'T', 'S']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G'] 
        self.opt.device = self.device
        self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.opt)
        
        print(self.netG)  

        if self.isTrain:
            util.saveprint(self.opt, 'netG', str(self.netG))  
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        

    def set_position(self, pos, patch_pos=None):
        b = self.opt.batch_size
        self.pixel_pos = pos.unsqueeze(0).repeat(b, 1, 1, 1).to(self.device)
        self.pixel_pos = self.pixel_pos.flatten(2).permute(2, 0, 1)
        if self.opt.pos_none:
            self.input_pos = None
        else:
            input_pos = self.PatchPositionEmbeddingSine(self.opt)
            self.input_pos = input_pos.unsqueeze(0).repeat(b, 1, 1, 1).to(self.device)
            self.input_pos = self.input_pos.flatten(2).permute(2, 0, 1)
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.image = input['image'].to(self.device, dtype=torch.float32)
        if self.isTrain:
            self.gt_AWB = input['gt-AWB'].to(self.device, dtype=torch.float32)
            self.gt_T = input['gt-T'].to(self.device, dtype=torch.float32)
            self.gt_S = input['gt-S'].to(self.device, dtype=torch.float32)
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
        self.AWB, self.T, self.S = self.netG(inputs=self.image, pixel_pos=self.input_pos)
    def compute_G_loss(self):
        """Calculate L1 loss for the generator"""
        self.loss_AWB_L1 = self.criterionL1(self.AWB, self.gt_AWB)*self.opt.lambda_L1
        self.loss_T_L1 = self.criterionL1(self.T, self.gt_T)*self.opt.lambda_L1
        self.loss_S_L1 = self.criterionL1(self.S, self.gt_S)*self.opt.lambda_L1
        self.loss_G = self.loss_AWB_L1 + self.loss_T_L1 + self.loss_S_L1
        return self.loss_G

    def optimize_parameters(self):
        # forward
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def PatchPositionEmbeddingSine(self, opt):
        temperature=10000
        if opt.stride == 1:
            feature_h = int(128/opt.ksize)
        else:
            feature_h = int((128-opt.ksize)/opt.stride)+1
        # feature_h = int(256/opt.ksize)*2
        num_pos_feats = 256//2
        mask = torch.ones((feature_h, feature_h))
        y_embed = mask.cumsum(0, dtype=torch.float32)
        x_embed = mask.cumsum(1, dtype=torch.float32)
        # if self.normalize:
        #     eps = 1e-6
        #     y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        #     x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        return pos
