import math

import torch.nn as nn

from .configuration_mdsr import MdsrConfig
from ...modeling_utils import (
    BamBlock,
    MeanShift,
    PreTrainedModel
)


class Upsampler(nn.Sequential):
    def __init__(self, scale, inp, bn=False, act=False, bias=True, choice=0):

        modules = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
           for _ in range(int(math.log(scale, 2))):
               modules.append(conv(inp, 4 * inp, 3, bias=bias))
               modules.append(nn.PixelShuffle(2))
               if bn: modules.append(nn.BatchNorm2d(inp))
               if act: modules.append(act())
        elif scale == 3:
           modules.append(conv(inp, 9 * inp, 3, bias=bias))
           modules.append(nn.PixelShuffle(3))
           if bn: modules.append(nn.BatchNorm2d(inp))
           if act: modules.append(act())
        else:
           raise NotImplementedError
        super(Upsampler, self).__init__(*modules)


def conv(inp, oup, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = ((kernel_size -1) * dilation + 1) // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
    )


class ResBlock(nn.Module):
    def __init__(self, inp, bam, kernel_size=3, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        modules = []
        for i in range(2):
            modules.append(conv(inp, inp, kernel_size, bias=bias))
            if bn: modules.append(nn.BatchNorm2d(inp))
            if i == 0: modules.append(act)

        self.body = nn.Sequential(*modules)
        self.res_scale = res_scale
        self.bam = bam
        if bam:
            self.attention = BamBlock(inp)

    def forward(self, x):
        if self.bam:
            res = self.attention(self.body(x)).mul(self.res_scale)
        else:
            res = self.body(x).mul(self.res_scale)
        res += x

        return res


class MdsrModel(PreTrainedModel):
    config_class = MdsrConfig

    def __init__(self, args):
        super(MdsrModel, self).__init__(args)

        # args
        self.scale_list = [args.scale]
        self.scale = args.scale
        bam = args.bam
        input_channel = 3
        output_channel = 3
        num_block = 32
        inp = 64
        rgb_range = 255
        res_scale = 0.1
        act = nn.ReLU(True)
        # act = nn.LeakyReLU(negative_slope=0.05, inplace=True)

        # head
        self.head = nn.Sequential(conv(3, inp, input_channel))

        # pre_process
        self.pre_process = nn.ModuleDict([str(scale),
                                          nn.Sequential(ResBlock(inp, bam, bias=True, act=act, res_scale=res_scale),
                                                        ResBlock(inp, bam, bias=True, act=act, res_scale=res_scale))]
                                         for scale in self.scale_list)

        # body
        self.body = nn.Sequential(
            *[ResBlock(inp, bam, bias=True, act=act, res_scale=res_scale) for _ in range(num_block)])
        self.body.add_module(str(num_block), conv(inp, inp, 3))

        # upsample
        self.upsample = nn.ModuleDict(
            [str(scale), Upsampler(scale, inp, act=False)] for scale in self.scale_list)

        # tail
        self.tail = nn.Sequential(conv(inp, 3, output_channel))

        self.sub_mean = MeanShift(rgb_range, sign=-1)
        self.add_mean = MeanShift(rgb_range, sign=1)

    def forward(self, x):
        scale_id = str(self.scale)
        # x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process[scale_id](x)

        res = self.body(x)
        res += x

        x = self.upsample[scale_id](res)
        x = self.tail(x)
        # x = self.add_mean(x)
        return x

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
