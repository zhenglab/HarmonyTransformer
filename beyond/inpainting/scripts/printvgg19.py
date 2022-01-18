import torch
import torch.nn as nn
import torchvision.models as models
import imp
# from vgg16_p365 import KitModel

vgg16 = models.vgg16(pretrained=True)
# vgg19 = models.vgg19(pretrained=True)
MainModel = imp.load_source('MainModel', "/media/disk/gds/code/outpainting/scripts/pytorch_caffe/data/vgg16_p365.py")
vgg16_ = torch.load("/media/disk/gds/code/outpainting/scripts/pytorch_caffe/data/vgg16_p365.pth")
vgg16_.eval()

print(vgg16)
print(vgg16_)

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers= extracted_layers

    def forward(self, x):
        outputs = []
        
        for name, module in self.submodule._modules.items():
            x = module(x)
            print(x.shape)
            print(name, self.extracted_layers, (name in self.extracted_layers))
            if name in self.extracted_layers:
                outputs += [x]
        return outputs

# vgg_fe = FeatureExtractor(vgg16_, ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])

_in = torch.randn(1, 3, 224, 224)
relus = vgg16_(_in)

print(_in.shape)

for i in range(5):
    print(relus[i].shape)
    
    