import torch
import torchvision
import torch.nn as nn
import numpy as np

##usage: add to train.py or test.py: misc.print_model_parm_nums(model)
##  misc.print_model_parm_flops(model,inputs)
def print_model_params(model):
    total = 0
#     total = sum([param.nelement() for param in model.parameters()])
    for name, param in model.named_parameters():
#         if "bn" not in name:
#             total += param.nelement()
        total += param.nelement()
    
    print('  + Number of params: %.4f(e6)' % (total / 1e6))

def print_model_flops(model, inputs=None, image=None, pixel_pos=None, patch_pos=None, mask_r=None, mask=None, layers=[], encode_only=False): #retinextr
# def print_model_flops(model, inputs, pos=None, mask_r=None, layers=[], encode_only=False, tgt_enc=None, ifm_image=None):
    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].shape
        output_channels, output_height, output_width = output[0].shape

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * 1
        bias_ops = 1 if self.bias is not None else 0
 
        params = output_channels * (kernel_ops + bias_ops)
        flops = 1 * params * output_height * output_width
        list_conv.append(flops)
        
    list_deconv=[]
    def deconv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].shape
        output_channels, output_height, output_width = output[0].shape
        
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * 1
        bias_ops = 1 if self.bias is not None else 0
 
        params = output_channels * (kernel_ops + bias_ops)
        flops = 1 * params * input_height * input_width
        list_deconv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = 8
 
        weight_ops = self.weight.nelement() * 1
        bias_ops = self.bias.nelement() if self.bias is not None else 0
 
        flops = 1 * (weight_ops + bias_ops)
        list_linear.append(flops)
 
    list_bn=[]
    def bn_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].shape
        list_bn.append(input[0].nelement())
     
    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_sigmoid=[]
    def sigmoid_hook(self, input, output):
        list_sigmoid.append(input[0].nelement())

    list_upsample=[]
    def upsample_hook(self, input, output):
        list_upsample.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].shape
        output_channels, output_height, output_width = output[0].shape
        
        kernel_ops = self.kernel_size[0] * self.kernel_size[1]
        bias_ops = 0
        
        params = output_channels * (kernel_ops + bias_ops)
        flops = 1 * params * output_height * output_width
        list_pooling.append(flops)
        
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(conv_hook)
        if isinstance(m, nn.ConvTranspose2d):
            m.register_forward_hook(deconv_hook)
        if isinstance(m, torch.nn.Linear):
            m.register_forward_hook(linear_hook)
        if isinstance(m, torch.nn.BatchNorm2d):
            m.register_forward_hook(bn_hook)
        # if isinstance(m, torch.nn.BatchNorm2d):
        #     m.register_forward_hook(bn_hook)
        if isinstance(m, torch.nn.ReLU):
            m.register_forward_hook(relu_hook)
        if isinstance(m, torch.nn.Upsample):
            m.register_forward_hook(upsample_hook)
        if isinstance(m, torch.nn.Sigmoid):
            m.register_forward_hook(sigmoid_hook)
        if isinstance(m, torch.nn.MaxPool2d) or isinstance(m, torch.nn.AvgPool2d):
            m.register_forward_hook(pooling_hook)
        
    output = model(inputs=inputs, image=image, pixel_pos=pixel_pos, patch_pos=patch_pos, mask_r=mask_r, mask=mask) #retinextr
    # output = model(inputs=inputs)
    total_flops = (sum(list_conv) + sum(list_deconv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_sigmoid) + sum(list_pooling))
    # total_flops = (sum(list_conv) + sum(list_deconv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_upsample))
    # total_flops = sum(list_linear)
    print('  + Number of FLOPs: %.6f(e9)' % (total_flops / 1e9))
