import functools

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .configuration_a2n import A2nConfig
from ...modeling_utils import (
    make_layer,
    PixelAttentionBlock,
    PreTrainedModel
)


class AttentionBranch(nn.Module):

    def __init__(self, nf, k_size=3):
        super(AttentionBranch, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution

    def forward(self, x):
        y = self.k1(x)
        y = self.lrelu(y)
        y = self.k2(y)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out


class A2Block(nn.Module):

    def __init__(self, nf, reduction=4, k=2, t=30):
        super(A2Block, self).__init__()
        self.t = t
        self.k = k

        self.conv_first = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.conv_last = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Attention Dropout Module
        self.ADM = nn.Sequential(
            nn.Linear(nf, nf // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf // reduction, self.k, bias=False),
        )

        # attention branch
        self.attention = AttentionBranch(nf)
        # non-attention branch
        # 3x3 conv for A2N
        self.non_attention = nn.Conv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        # 1x1 conv for A2N-M
        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        a, b, c, d = x.shape

        x = self.conv_first(x)
        x = self.lrelu(x)

        # Attention Dropout
        y = self.avg_pool(x).view(a, b)
        y = self.ADM(y)
        ax = functional.softmax(y / self.t, dim=1)

        attention = self.attention(x)
        non_attention = self.non_attention(x)

        x = attention * ax[:, 0].view(a, 1, 1, 1) + non_attention * ax[:, 1].view(a, 1, 1, 1)
        x = self.lrelu(x)

        out = self.conv_last(x)
        out += residual

        return out


class A2nModel(PreTrainedModel):
    config_class = A2nConfig

    def __init__(self, args):
        super(A2nModel, self).__init__(args)

        in_nc = args.in_nc
        out_nc = args.out_nc
        nf = args.nf
        unf = args.unf
        nb = args.nb
        scale = args.scale

        # AAB
        AAB_block_f = functools.partial(A2Block, nf=nf)
        self.scale = scale

        # first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        # main blocks
        self.AAB_trunk = make_layer(AAB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PixelAttentionBlock(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PixelAttentionBlock(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.AAB_trunk(fea))
        fea = fea + trunk

        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(functional.interpolate(fea, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(functional.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(functional.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))

        out = self.conv_last(fea)

        ilr = functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ilr

        return out

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError(f'While copying the parameter named {name}, '
                                           f'whose dimensions in the model are {own_state[name].size()} and '
                                           f'whose dimensions in the checkpoint are {param.size()}.')
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(f'unexpected key "{name}" in state_dict')
