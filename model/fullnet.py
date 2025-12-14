import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import *
from .sparsest import Sparsest

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

class FullNet(nn.Module):
    def __init__(self, in_channel_event, in_channel_image, out_channels, mid_channels, sparse_channels, padding_mode='reflect', bias=False, basic_deform=False, deform_q=False, deform_kv=True):
        super(FullNet, self).__init__()
        self.sparseest = Sparsest(in_channel_event, sparse_channels)
        self.resnet_event = MyFusionMultiResNet(sparse_channels, mid_channels, stem=False, maxpool=True, basicdeform=basic_deform)
        self.resnet_image = MyFusionMultiResNet(in_channel_image, mid_channels, stem=True, maxpool=False)

        self.refine = MultiDepthRefine(padding_mode=padding_mode, bias=bias, deform_q=deform_q, deform_kv=deform_kv)
        self.conv1 = nn.Conv2d(mid_channels, out_channels, 7,1,3, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,1,1, bias=bias)
        self.softmax = CustomSoftmax()

    def forward(self, x_event, x_image, valid_mask=None):
        sparse = self.sparseest(x_event)
        sparse_full = sparse.clone()
        # print(sparse.shape)
        # print(valid_mask.shape)
        valid_mask = valid_mask.unsqueeze(1).expand_as(sparse)
        if valid_mask is not None:
            sparse = sparse * valid_mask
        x_event = self.resnet_event(sparse)
        x_image = self.resnet_image(x_image)

        # x = self.crossattn(x_image, x_event)
        outputs = self.refine(x_event, x_image)
        x = self.conv1(outputs[-1]+x_image[-1]+x_event[-1])
        x = F.relu(x)
        x = self.conv2(x)
        x = self.softmax(x)
        return x, sparse_full, outputs

