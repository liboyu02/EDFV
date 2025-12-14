from torchvision.models.resnet import ResNet, Bottleneck
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Type, Union
from .deformconv import DeformConv2d
from einops import rearrange
from .blocks import *

def _make_fusion_block(features, use_bn, size=None, expand=True):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=expand,
        align_corners=True,
        size=size,
    )

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MCC(nn.Module):
    def __init__(self, f_number, num_heads, padding_mode='reflect', bias=False, deform_q=True, deform_kv=True) -> None:
        super().__init__()
        self.norm = LayerNorm(f_number, eps=1e-6, data_format='channels_first')

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.pwconv_q = nn.Conv2d(f_number, f_number , kernel_size=1, bias=bias)
        if deform_q:
            self.dwconv_q = DeformConv2d(f_number,f_number, kernel_size=3, padding=1, stride=1, bias=bias, padding_mode=padding_mode)  
        else:
            self.dwconv_q = nn.Conv2d(f_number , f_number , 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=f_number)

        self.pwconv_kv = nn.Conv2d(f_number, f_number * 2, kernel_size=1, bias=bias)
        
        if deform_kv:
            self.dwconv_kv = DeformConv2d(f_number*2, f_number * 2, kernel_size=3, padding=1, stride=1, bias=bias, padding_mode=padding_mode)
        else:
            self.dwconv_kv = nn.Conv2d(f_number * 2, f_number * 2, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=f_number * 2)

        self.project_out = nn.Conv2d(f_number, f_number, kernel_size=1, bias=bias)
        self.feedforward = nn.Sequential(
            nn.Conv2d(f_number, f_number, 1, 1, 0, bias=bias),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(f_number, f_number, 3, 1, 1, bias=bias, groups=f_number, padding_mode=padding_mode),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x_Q, x_KV):
        attn_q = self.norm(x_Q)
        attn_kv = self.norm(x_KV)
        _, _, h, w = attn_q.shape

        q = self.dwconv_q(attn_q)
        k, v = self.dwconv_kv(self.pwconv_kv(attn_kv)).chunk(2, dim=1)
        # qkv = self.dwconv(self.pwconv(attn))
        # q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.feedforward(out+x_Q)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, deform=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if deform:
                self.shortcut = nn.Sequential(
                    DeformConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    # nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MyFusionMultiResNet(nn.Module):
    def __init__(self, in_channels=1, output_channels=1,norm_layer: Optional[Callable[..., nn.Module]] = None,stem=True, maxpool=False, basicdeform=False, use_bn=True):
        super(MyFusionMultiResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.stem_bool = stem
        self.maxpool_bool = maxpool
        assert not (self.stem_bool and self.maxpool_bool), "Cannot set both stem and maxpool to True"

        if stem:
            self.inplanes = 8
            self.conv0 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.inplanes = in_channels

        self.layer1 = BasicBlock(self.inplanes, 16, stride=1, deform=basicdeform)
        self.layer2 = BasicBlock(16, 32, stride=2, deform=basicdeform)
        self.layer3 = BasicBlock(32, 64, stride=2, deform=basicdeform)
        self.layer4 = BasicBlock(64, 128, stride=2, deform=basicdeform)

        self.fuse4 = _make_fusion_block(128,use_bn,expand=True)
        self.fuse3 = _make_fusion_block(64,use_bn,expand=True)
        self.fuse2 = _make_fusion_block(32,use_bn,expand=True)

        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.stem_bool or self.maxpool_bool:
            self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv4 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
            out_channels = 8    
        else:
            out_channels = 16

            # self.upsample5 = nn.Upsample(scale_factor=2, mode='nearest')
            # self.conv5 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)

        # self.conv_output = nn.Conv2d(out_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if self.stem_bool:
            x = self.conv0(x)
            x = self.bn1(x)
            x = self.relu(x)
            # x = self.maxpool(x)
        
        if self.maxpool_bool:
            x = self.maxpool(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        
        path4 = self.fuse4(out4, size=out3.shape[-2:])
        path3 = self.fuse3(path4, out3,size=out2.shape[-2:])
        path2 = self.fuse2(path3, out2,size=out1.shape[-2:])
        out = path2

        if self.stem_bool or self.maxpool_bool:
            out = self.upsample4(out)
            out = F.relu(self.conv4(out))

        # out = self.conv_output(out)
        outputs = [path4, path3, path2, out]
        return outputs

class MultiDepthRefine(nn.Module):
    def __init__(self, in_channels=[64, 32,16, 8], use_bn=True, padding_mode='reflect', bias=False, deform_q=False, deform_kv=True):
        super(MultiDepthRefine, self).__init__()
        self.in_channels = in_channels
        self.mccs = nn.ModuleList()
        self.fuses = nn.ModuleList()
        for i in range(len(in_channels)):
            self.mccs.append(MCC(in_channels[i], in_channels[i]//4, padding_mode=padding_mode, bias=bias, deform_q=deform_q, deform_kv=deform_kv))
            if i < len(in_channels)-1:
                self.fuses.append(_make_fusion_block(in_channels[i],use_bn,expand=True))
            else:
                self.fuses.append(_make_fusion_block(in_channels[i],use_bn,expand=False))
    
    def forward(self, x_event, x_image):
        crosses = []
        for i in range(4):
            crosses.append(self.mccs[i](x_image[i], x_event[i]))
        path0 = self.fuses[0](crosses[0], size=crosses[1].shape[-2:])
        path1 = self.fuses[1](path0, crosses[1],size=crosses[2].shape[-2:])
        path2 = self.fuses[2](path1, crosses[2],size=crosses[3].shape[-2:])
        path3 = self.fuses[3](path2, crosses[3],size=crosses[3].shape[-2:])
        out = path3
        outputs = [path0, path1, path2, path3]
        return outputs

class AblationNaiveMultiDepthRefine(nn.Module):
    def __init__(self, in_channels=[64, 32,16, 8], use_bn=True, padding_mode='reflect', bias=False, deform_q=False, deform_kv=True):
        super(AblationNaiveMultiDepthRefine, self).__init__()
        self.in_channels = in_channels
        # self.mccs = nn.ModuleList()
        self.fuses = nn.ModuleList()
        for i in range(len(in_channels)):
            # self.mccs.append(MCC(in_channels[i], in_channels[i]//4, padding_mode=padding_mode, bias=bias, deform_q=deform_q, deform_kv=deform_kv))
            if i < len(in_channels)-1:
                self.fuses.append(_make_fusion_block(in_channels[i],use_bn,expand=True))
            else:
                self.fuses.append(_make_fusion_block(in_channels[i],use_bn,expand=False))
    
    def forward(self, x_event, x_image):
        crosses = []
        for i in range(4):
            # crosses.append(self.mccs[i](x_image[i], x_event[i]))
            crosses.append(x_image[i]+x_event[i])
        path0 = self.fuses[0](crosses[0], size=crosses[1].shape[-2:])
        path1 = self.fuses[1](path0, crosses[1],size=crosses[2].shape[-2:])
        path2 = self.fuses[2](path1, crosses[2],size=crosses[3].shape[-2:])
        path3 = self.fuses[3](path2, crosses[3],size=crosses[3].shape[-2:])
        out = path3
        outputs = [path0, path1, path2, path3]
        return outputs

class AblationNaiveMultiDepthRefineV2(nn.Module):
    def __init__(self, in_channels=[64, 32,16, 8], use_bn=True, padding_mode='reflect', bias=False, deform_q=False, deform_kv=True):
        super(AblationNaiveMultiDepthRefineV2, self).__init__()
        self.in_channels = in_channels
        # self.mccs = nn.ModuleList()
        # self.fuses = nn.ModuleList()
        # for i in range(len(in_channels)):
            # self.mccs.append(MCC(in_channels[i], in_channels[i]//4, padding_mode=padding_mode, bias=bias, deform_q=deform_q, deform_kv=deform_kv))
            # if i < len(in_channels)-1:
            #     self.fuses.append(_make_fusion_block(in_channels[i],use_bn,expand=True))
            # else:
            #     self.fuses.append(_make_fusion_block(in_channels[i],use_bn,expand=False))
    
    def forward(self, x_event, x_image):
        crosses = []
        for i in range(4):
            # crosses.append(self.mccs[i](x_image[i], x_event[i]))
            crosses.append(x_image[i]+x_event[i])
        path0 = self.fuses[0](crosses[0], size=crosses[1].shape[-2:])
        path1 = self.fuses[1](path0, crosses[1],size=crosses[2].shape[-2:])
        path2 = self.fuses[2](path1, crosses[2],size=crosses[3].shape[-2:])
        path3 = self.fuses[3](path2, crosses[3],size=crosses[3].shape[-2:])
        out = path3
        outputs = [path0, path1, path2, path3]
        return outputs

class AblationMultiDepthRefine(nn.Module):
    def __init__(self, in_channels=[64, 32,16, 8], use_bn=True, padding_mode='reflect', bias=False, deform_q=False, deform_kv=True):
        super(AblationMultiDepthRefine, self).__init__()
        self.in_channels = in_channels
        # self.mccs = nn.ModuleList()
        self.fuses = nn.ModuleList()
        for i in range(len(in_channels)):
            # self.mccs.append(MCC(in_channels[i], in_channels[i]//4, padding_mode=padding_mode, bias=bias, deform_q=deform_q, deform_kv=deform_kv))
            if i < len(in_channels)-1:
                self.fuses.append(_make_fusion_block(in_channels[i],use_bn,expand=True))
            else:
                self.fuses.append(_make_fusion_block(in_channels[i],use_bn,expand=False))
    
    def forward(self, x_image):
        # crosses = []
        # for i in range(4):
        #     crosses.append(self.mccs[i](x_image[i], x_event[i]))
        path0 = self.fuses[0](x_image[0], size=x_image[1].shape[-2:])
        path1 = self.fuses[1](path0, x_image[1],size=x_image[2].shape[-2:])
        path2 = self.fuses[2](path1, x_image[2],size=x_image[3].shape[-2:])
        path3 = self.fuses[3](path2, x_image[3],size=x_image[3].shape[-2:])
        out = path3
        outputs = [path0, path1, path2, path3]
        return outputs        


class MyFusionResNet(nn.Module):
    def __init__(self, in_channels=1, output_channels=1,norm_layer: Optional[Callable[..., nn.Module]] = None,stem=True, maxpool=False, basicdeform=False, use_bn=True):
        super(MyFusionResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.stem_bool = stem
        self.maxpool_bool = maxpool
        assert not (self.stem and self.maxpool), "Cannot set both stem and maxpool to True"

        if stem:
            self.inplanes = 8
            self.conv0 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.inplanes = in_channels

        self.layer1 = BasicBlock(self.inplanes, 16, stride=1, deform=basicdeform)
        self.layer2 = BasicBlock(16, 32, stride=2, deform=basicdeform)
        self.layer3 = BasicBlock(32, 64, stride=2, deform=basicdeform)
        self.layer4 = BasicBlock(64, 128, stride=2, deform=basicdeform)

        self.fuse4 = _make_fusion_block(128,use_bn,expand=True)
        self.fuse3 = _make_fusion_block(64,use_bn,expand=True)
        self.fuse2 = _make_fusion_block(32,use_bn,expand=True)

        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.stem_bool or self.maxpool_bool:
            self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv4 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
            out_channels = 8    
        else:
            out_channels = 16

            # self.upsample5 = nn.Upsample(scale_factor=2, mode='nearest')
            # self.conv5 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)

        self.conv_output = nn.Conv2d(out_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if self.stem_bool:
            x = self.conv0(x)
            x = self.bn1(x)
            x = self.relu(x)
            # x = self.maxpool(x)
        
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        
        path4 = self.fuse4(out4, size=out3.shape[-2:])
        path3 = self.fuse3(path4, out3,size=out2.shape[-2:])
        path2 = self.fuse2(path3, out2,size=out1.shape[-2:])
        out = path2

        if self.maxpool_bool:
            out = self.maxpool(out)

        if self.stem_bool or self.maxpool_bool:
            out = self.upsample4(out)
            out = F.relu(self.conv4(out))

        out = self.conv_output(out)
        
        return out

class MyResNet(nn.Module):
    def __init__(self, in_channels=1, output_channels=1,norm_layer: Optional[Callable[..., nn.Module]] = None,stem=True, maxpool=False, basicdeform=False):
        super(MyResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.stem = stem
        self.maxpool = maxpool
        assert not (self.stem and self.maxpool), "Cannot set both stem and maxpool to True"

        if stem:
            self.inplanes = 8
            self.conv0 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.inplanes = in_channels

        self.layer1 = BasicBlock(self.inplanes, 16, stride=1, deform=basicdeform)
        self.layer2 = BasicBlock(16, 32, stride=2, deform=basicdeform)
        self.layer3 = BasicBlock(32, 64, stride=2, deform=basicdeform)
        self.layer4 = BasicBlock(64, 128, stride=2, deform=basicdeform)

        # self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.conv_transpose3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)

        if self.maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.stem or self.maxpool:
            self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv4 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
            out_channels = 8    
        else:
            out_channels = 16

            # self.upsample5 = nn.Upsample(scale_factor=2, mode='nearest')
            # self.conv5 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)

        self.conv_output = nn.Conv2d(out_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if self.stem:
            x = self.conv0(x)
            x = self.bn1(x)
            x = self.relu(x)
            # x = self.maxpool(x)
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 使用转置卷积层进行上采样
        # out = self.conv_transpose1(out)
        # out = self.conv_transpose2(out)
        # out = self.conv_transpose3(out)
        out = self.upsample1(out)
        out = F.relu(self.conv1(out))
        
        out = self.upsample2(out)
        out = F.relu(self.conv2(out))
        
        out = self.upsample3(out)
        out = F.relu(self.conv3(out))

        if self.maxpool:
            out = self.maxpool(out)

        if self.stem or self.maxpool:
            out = self.upsample4(out)
            out = F.relu(self.conv4(out))

            # out = self.upsample5(out)
            # out = F.relu(self.conv5(out))

        out = self.conv_output(out)
        
        return out