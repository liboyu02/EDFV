import torch
import torch.nn as nn

import torch.nn.functional as F
from .resnet import BasicBlock

def get_norm(norm='none'):
    if norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif norm == 'layer':
        norm_layer = nn.LayerNorm
    elif norm == 'none':
        norm_layer = nn.Identity
    else:
        print("=====Wrong norm type!======")
    return norm_layer

def conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1, norm = 'none'):
    norm_layer = get_norm(norm)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,stride=stride,padding=padding),
        norm_layer(out_ch),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )

class downConv(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.res = BasicBlock(in_ch, in_ch, stride=1)
        self.conv2 = conv(in_ch, in_ch, kernel_size=1,stride=1,padding=0)
        self.down = conv(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.res(x)
        # x_skip = self.conv2(x)
        x_skip = self.conv2(x)
        x = self.down(x)
        return x, x_skip

class upConv_v2(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch) -> None:
        super().__init__()
        self.conv1 = conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv(out_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x, x_skip):
        #x = self.deconv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        #x_skip = self.conv_skip(x_skip)
        # print(x.shape, x_skip.shape)
        x = torch.cat([x, x_skip], dim=1)     
        x = self.conv2(x)
        
        return x

class CustomSoftmax(nn.Module):
    def __init__(self):
        super(CustomSoftmax, self).__init__()
        self.softplus = nn.Softplus()

    def forward(self, x):
        # Apply the Softplus function
        softplus_output = self.softplus(x)
        # Normalize to get the probability distribution
        sum_softplus_output = torch.sum(softplus_output, dim=1, keepdim=True)
        output = softplus_output / sum_softplus_output
        return output

class Sparsest(nn.Module):
    def __init__(self, channels=64, out_ch = 64, base_chs = 32, depth = 3):
        super(Sparsest, self).__init__()

        self.depth = depth
        self.head = nn.Conv2d(channels, base_chs, kernel_size=3, stride=1, padding=1)

        self.EDFV_encoder = nn.ModuleList()
        self.EDFV_decoder = nn.ModuleList()

        for i in range(self.depth):
            self.EDFV_encoder.append(downConv(base_chs*2**i, base_chs*2**(i+1)))
        
        self.bottom = nn.Sequential(
            BasicBlock(base_chs*2**self.depth, base_chs*2**self.depth, stride=1),
            BasicBlock(base_chs*2**self.depth, base_chs*2**self.depth, stride=1),
        )

        for i in range(1,self.depth+1):
            self.EDFV_decoder.append(upConv_v2(base_chs*2**i, base_chs*2**(i-1), base_chs*2**(i-1)))

        self.pred = nn.Conv2d(base_chs, out_ch, kernel_size=3, stride=1, padding=1)
        self.softmax = CustomSoftmax()

    def forward(self, x):
        
        x = self.head(x)
        
        EDFV_list = []
        for i in range(self.depth):
            x, EDFV = self.EDFV_encoder[i](x)
            
            EDFV_list.append(EDFV)
        x = self.bottom(x)
        
        for i in range(self.depth-1, -1, -1):
            
            x = self.EDFV_decoder[i](x, EDFV_list[i])
        x = self.pred(x)
        x = self.softmax(x)
        return x