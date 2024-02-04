import math
import numpy as np

import torch
import torch.nn as nn

from .configuration_physicssr import PhysicssrConfig
from ...modeling_utils import (
    default_conv,
    MeanShift,
    Upsampler,
    PreTrainedModel
)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class PhysicssrModel(PreTrainedModel):
    config_class = PhysicssrConfig

    def __init__(self, args, conv=default_conv):
        super(PhysicssrModel, self).__init__(args)

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        act = nn.ReLU(True)

        rgb_mean = args.rgb_mean
        rgb_std = args.rgb_std
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        # define refinement module
        m_tail_refine = [
            conv(n_feats, args.n_colors, kernel_size)
        ]
        self.head_refine1 = nn.Sequential(*m_head)
        self.body_refine1 = nn.Sequential(*m_body)
        self.tail_refine1 = nn.Sequential(*m_tail_refine)

        self.head_refine2 = nn.Sequential(*m_head)
        self.body_refine2 = nn.Sequential(*m_body)
        self.tail_refine2 = nn.Sequential(*m_tail_refine)

        # gaussian blur operator and pixel substitution operator
        self.scale = args.scale
        gaussian_blur_sigma = 0
        if self.scale == 2:
            gaussian_blur_sigma = 0.5  # 0.4~0.6
        elif self.scale == 3:
            gaussian_blur_sigma = 0.9  # 0.8~1.0
        elif self.scale == 4:
            gaussian_blur_sigma = 1.3  # 1.2~1.4
        if gaussian_blur_sigma == 0:
            raise (RuntimeError("gaussian_blur_sigma = 0!"))
        gaussian_blur_kernel_size = int(math.ceil(gaussian_blur_sigma * 3) * 2 + 1)
        gaussian_blur_kernel = self.matlab_style_gauss2D(shape=(gaussian_blur_kernel_size, gaussian_blur_kernel_size),
                                                         sigma=gaussian_blur_sigma)
        self.gaussian_blur = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=gaussian_blur_kernel_size, stride=1,
                                       padding=int((gaussian_blur_kernel_size - 1) / 2), bias=False)
        nn.init.constant_(self.gaussian_blur.weight.data, 0.0)
        self.gaussian_blur.weight.data[0, 0, :, :] = torch.FloatTensor(gaussian_blur_kernel)
        self.gaussian_blur.weight.data[1, 1, :, :] = torch.FloatTensor(gaussian_blur_kernel)
        self.gaussian_blur.weight.data[2, 2, :, :] = torch.FloatTensor(gaussian_blur_kernel)

    def matlab_style_gauss2D(self, shape=(5, 5), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def pixel_substitution(self, x, x_refine):
        x_refine[:, :, np.tile(range(0, x_refine.shape[2], self.scale), (x.shape[3], 1)).T,
        np.tile(range(0, x_refine.shape[3], self.scale), (x.shape[2], 1))] = x
        return x_refine

    def forward(self, x):
        # the coarsest phase
        x_coarsest = (x.clone()).detach()
        x_coarsest = self.sub_mean(x_coarsest)
        x_coarsest = self.head(x_coarsest)

        res_coarsest = self.body(x_coarsest)
        res_coarsest += x_coarsest

        x_coarsest = self.tail(res_coarsest)
        x_coarsest = self.add_mean(x_coarsest)

        # refinement 1
        x_refine1 = (x_coarsest.clone()).detach()
        x_refine1 = self.gaussian_blur(x_refine1)
        x_refine1 = self.pixel_substitution(x, x_refine1)
        x_refine1 = x_refine1.detach()

        x_refine1 = self.sub_mean(x_refine1)
        x_refine1 = self.head_refine1(x_refine1)

        res_refine1 = self.body_refine1(x_refine1)
        res_refine1 += x_refine1

        x_refine1 = self.tail_refine1(res_refine1)
        x_refine1 = self.add_mean(x_refine1)

        # refinement 2
        x_refine2 = (x_refine1.clone()).detach()
        x_refine2 = self.gaussian_blur(x_refine2)
        x_refine2 = self.pixel_substitution(x, x_refine2)
        x_refine2 = x_refine2.detach()

        x_refine2 = self.sub_mean(x_refine2)
        x_refine2 = self.head_refine2(x_refine2)

        res_refine2 = self.body_refine2(x_refine2)
        res_refine2 += x_refine2

        x_refine2 = self.tail_refine2(res_refine2)
        x_refine2 = self.add_mean(x_refine2)

        return x_coarsest, x_refine1, x_refine2

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name == 'gaussian_blur.weight':
                continue
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
