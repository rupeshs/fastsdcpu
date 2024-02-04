import torch
import torch.nn as nn

from .configuration_awsrn import AwsrnConfig
from ...modeling_utils import (
    BamBlock,
    PreTrainedModel
)


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class AWRU(nn.Module):
    def __init__(
            self, n_feats, kernel_size, block_feats, wn, res_scale=1, act=nn.ReLU(True)):
        super(AWRU, self).__init__()
        self.res_scale = Scale(res_scale)
        self.x_scale = Scale(1)
        body = []
        body.append(
            wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size // 2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size // 2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.res_scale(self.body(x)) + self.x_scale(x)
        return res


class AWMS(nn.Module):
    def __init__(
            self, args, scale, n_feats, kernel_size, wn):
        super(AWMS, self).__init__()
        out_feats = scale * scale * 3
        self.tail_k3 = wn(nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2, dilation=1))
        self.tail_k5 = wn(nn.Conv2d(n_feats, out_feats, 5, padding=5 // 2, dilation=1))
        self.tail_k7 = wn(nn.Conv2d(n_feats, out_feats, 7, padding=7 // 2, dilation=1))
        self.tail_k9 = wn(nn.Conv2d(n_feats, out_feats, 9, padding=9 // 2, dilation=1))
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.scale_k3 = Scale(0.25)
        self.scale_k5 = Scale(0.25)
        self.scale_k7 = Scale(0.25)
        self.scale_k9 = Scale(0.25)

    def forward(self, x):
        x0 = self.pixelshuffle(self.scale_k3(self.tail_k3(x)))
        x1 = self.pixelshuffle(self.scale_k5(self.tail_k5(x)))
        x2 = self.pixelshuffle(self.scale_k7(self.tail_k7(x)))
        x3 = self.pixelshuffle(self.scale_k9(self.tail_k9(x)))

        return x0 + x1 + x2 + x3


class LFB(nn.Module):
    def __init__(
            self, n_feats, kernel_size, block_feats, n_awru, wn, res_scale, act=nn.ReLU(True)):
        super(LFB, self).__init__()
        self.n = n_awru
        self.lfl = nn.ModuleList([AWRU(n_feats, kernel_size, block_feats, wn=wn, res_scale=res_scale, act=act)
                                  for i in range(self.n)])

        self.reduction = wn(nn.Conv2d(n_feats * self.n, n_feats, kernel_size, padding=kernel_size // 2))

        self.res_scale = Scale(res_scale)
        self.x_scale = Scale(1)

    def forward(self, x):
        s = x
        out = []
        for i in range(self.n):
            x = self.lfl[i](x)
            out.append(x)
        res = self.reduction(torch.cat(out, dim=1))
        return self.res_scale(res) + self.x_scale(s)


class AwsrnModel(PreTrainedModel):
    config_class = AwsrnConfig

    def __init__(self, args):
        super(AwsrnModel, self).__init__(args)
        # hyper-params
        self.args = args
        self.bam = args.bam
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        scale = args.scale
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        res_scale = args.res_scale
        n_awru = args.n_awru
        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(args.rgb_mean)).view([1, 3, 1, 1])

        # define head module
        # head = HEAD(args, n_feats, kernel_size, wn)
        head = []
        head.append(
            wn(nn.Conv2d(args.n_colors, n_feats, 3, padding=3 // 2)))

        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(
                LFB(n_feats, kernel_size, args.block_feats, n_awru, wn=wn, res_scale=res_scale, act=act))

        # define tail module
        out_feats = scale * scale * args.n_colors
        tail = AWMS(args, scale, n_feats, kernel_size, wn)

        skip = []
        skip.append(
            wn(nn.Conv2d(args.n_colors, out_feats, 3, padding=3 // 2))
        )
        skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = tail
        self.skip = nn.Sequential(*skip)
        if self.bam:
            self.attention = BamBlock(args.block_feats // 4, reduction=8)

    def forward(self, x):
        if not self.bam:
            x = (x - self.rgb_mean.to(self.device) * 255) / 127.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        if self.bam:
            x = self.attention(x)
        x = self.tail(x)
        x += s
        if not self.bam:
            x = x * 127.5 + self.rgb_mean.to(self.device) * 255
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
                    if name.find('tail') >= 0 or  name.find('skip') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
