import torch
import torch.nn as nn
import functools
from torch.autograd import Variable

class Reshape(nn.Module):
    def __init__(self, fin, fout):
        super(Reshape, self).__init__()

        self.f_q = nn.Conv2d(fin, fout, kernel_size=1)
        self.f_k = nn.Conv2d(fin, fout, kernel_size=1)
        self.f_v = nn.Conv2d(fin, fin, kernel_size=1)

        self.fout= fout

        # self.max_pool = nn.MaxPool2d(2)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, f, mask):
        (b, c, h, w) = f.shape
        c_shrink = self.fout

        # f_in
        f_in = f

        # mask crop
        f = f.view(b, c, -1)
        mask_one_channel = mask.view(b, 1, -1)[0][0]
        
        index_outside = torch.nonzero(1 - mask_one_channel)
        index_inside = torch.nonzero(mask_one_channel)

        f_outside = f[:, :, index_outside]
        f_inside = f[:, :, index_inside]

        # outside => query
        f_q = self.f_q(f_outside).view(b, c_shrink, -1)

        # inside => key
        f_k = self.f_k(f_inside).view(b, c_shrink, -1)
        
        # inside => value
        f_v = self.f_v(f_inside).view(b, c, -1)

        # attention => query x key
        energy = torch.bmm(f_q.permute(0, 2, 1), f_k)
        att = self.softmax(energy)

        # reshape value => value x attention
        r_v = torch.bmm(f_v, att.permute(0, 2, 1))

        # paste r_v to f
        f[:, :, index_outside] = r_v.unsqueeze(-1)

        # f (b, c, -1) => f (b, c, h, w)
        f_reshape = f.view(b, c, h, w)

        # f_out
        f_out = f_reshape * (1 - mask) * self.gamma + f_in

        return f_reshape, f_out
