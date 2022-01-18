import torch
import torch.nn as nn
import functools
from torch.autograd import Variable

class Reshape(nn.Module):
    def __init__(self, fin):
        super(Reshape, self).__init__()

        nhidden = 128
        # self.max_pool = nn.MaxPool2d(2)
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.conv_w1 = nn.Conv2d(fin, fin, 1, padding=0)
        self.conv_w2 = nn.Conv2d(fin, fin, 1, padding=0)
        # self.norm = nn.InstanceNorm2d(fin)
        self.relu = nn.LeakyReLU(0.2, False)
        # self.mlp_shared = nn.Sequential(
        #     nn.Conv2d(fin, nhidden, kernel_size=1, padding=0),
        #     nn.ReLU()
        # )
        # self.mlp_gamma = nn.Conv2d(nhidden, fin, kernel_size=1, padding=0)
        # self.mlp_beta = nn.Conv2d(nhidden, fin, kernel_size=1, padding=0)

        # self.down = nn.Conv2d(fin, fin//16, 1, padding=0)
        self.down = nn.Upsample(scale_factor=0.5)
        self.up = nn.Upsample(scale_factor=2)
    
    def forward(self, f, mask):
        # f_in
        f_in = f
        mask_in = mask

        # mask crop
        f = self.down(f)
        mask = self.down(mask)
        
        (b, c, h, w) = f.shape

        f = f.view(b, c, -1)
        mask_one_channel = mask.view(b, 1, -1)[0][0]
        
        index_outside = torch.nonzero(1 - mask_one_channel)
        index_inside = torch.nonzero(mask_one_channel)

        # f_outside = f[:, :, index_outside] # 4, 128, 12288, 1
        # f_inside = f[:, :, index_inside] # 4, 128, 4096, 1

        # f_o = f_outside.expand(f_ous)

        o_re = index_outside.shape[0]
        i_re = index_inside.shape[0]
        # print(o_re, i_re)
        f_outside = f[:, :, index_outside] # 4, 128, 12288, 1
        f_inside = f[:, :, index_inside] # 4, 128, 4096, 1
        # f_outside = torch.cat([f_outside]*i_re, dim = 3)
        # f_inside = torch.cat([f_inside]*o_re, dim = 3)

        # 28 for test
        # f_o = f_outside.expand(b, c, o_re, i_re)
        # f_i = f_inside.expand(b, c, i_re, o_re)

        # # b, 1, o, 1
        # io_abs = torch.abs(f_o - f_i.permute(0, 1, 3, 2))

        # 29 for train and test
        # cosine
        f_o = f_outside.view(b, c, -1)
        f_i = f_inside.view(b, c, -1)
        matmul = torch.bmm(f_i.permute(0, 2, 1), f_o)
        f_i_abs = torch.sqrt(torch.sum(f_i.pow(2) + 1e-6, dim=1, keepdim=True))
        f_o_abs = torch.sqrt(torch.sum(f_o.pow(2) + 1e-6, dim=1, keepdim=True))
        abs_matmul = torch.bmm(f_i_abs.permute(0, 2, 1), f_o_abs)
        io_abs = matmul / abs_matmul
        # print(io_abs.shape)

        # print(torch.max(io_abs), torch.min(io_abs))
        
        # 28 for train and test
        # _map = torch.argmin(torch.sum(io_abs, dim=1, keepdim=True), dim=3).view(b, o_re)
        _map = torch.argmax(io_abs, dim=1)
        # print(_map.shape)
        # .view(b, o_re)
        
        f_oo = f_outside
        for i in range(b):
            f_oo[i] = f_inside[i, :, _map[i], :]
        
        f[:, :, index_outside] = f_oo
        f_out = f.view(b, c, h, w)

        f_out = self.up(f_out)

        # f_final = self.conv(torch.cat((f_out, f_in), dim=1))
        f_final = self.conv_w1(f_out) + self.conv_w2(f_in)
        # f_final = f_final * (1 - mask_in) + f_in * mask_in
        # f_final = self.norm(f_final)
        f_final = self.relu(f_final)
        # f_mlp = self.mlp_shared(f_out)
        # gamma = self.mlp_gamma(f_mlp)
        # beta = self.mlp_beta(f_mlp)

        # f_final = f_in * (1 + gamma) + beta

        return f_final, f_out