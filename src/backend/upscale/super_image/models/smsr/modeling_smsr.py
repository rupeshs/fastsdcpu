import torch
import torch.nn as nn
import torch.nn.functional as functional

from .configuration_smsr import SmsrConfig
from ...modeling_utils import (
    default_conv,
    MeanShift,
    PreTrainedModel
)


def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)

        return x * y


class SMB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, n_layers=4):
        super(SMB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        self.tau = 1
        self.relu = nn.ReLU(True)

        # channels mask
        self.ch_mask = nn.Parameter(torch.rand(1, out_channels, n_layers, 2))

        # body
        body = []
        body.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        for _ in range(self.n_layers - 1):
            body.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias))
        self.body = nn.Sequential(*body)

        # collect
        self.collect = nn.Conv2d(out_channels * self.n_layers, out_channels, 1, 1, 0)

    def _update_tau(self, tau):
        self.tau = tau

    def _prepare(self):
        # channel mask
        ch_mask = self.ch_mask.softmax(3).round()
        self.ch_mask_round = ch_mask

        # number of channels
        self.d_in_num = []
        self.s_in_num = []
        self.d_out_num = []
        self.s_out_num = []

        for i in range(self.n_layers):
            if i == 0:
                self.d_in_num.append(self.in_channels)
                self.s_in_num.append(0)
                self.d_out_num.append(int(ch_mask[0, :, i, 0].sum(0)))
                self.s_out_num.append(int(ch_mask[0, :, i, 1].sum(0)))
            else:
                self.d_in_num.append(int(ch_mask[0, :, i - 1, 0].sum(0)))
                self.s_in_num.append(int(ch_mask[0, :, i - 1, 1].sum(0)))
                self.d_out_num.append(int(ch_mask[0, :, i, 0].sum(0)))
                self.s_out_num.append(int(ch_mask[0, :, i, 1].sum(0)))

        # kernel split
        kernel_d2d = []
        kernel_d2s = []
        kernel_s = []

        for i in range(self.n_layers):
            if i == 0:
                kernel_s.append([])
                if self.d_out_num[i] > 0:
                    kernel_d2d.append(self.body[i].weight[ch_mask[0, :, i, 0] == 1, ...].view(self.d_out_num[i], -1))
                else:
                    kernel_d2d.append([])
                if self.s_out_num[i] > 0:
                    kernel_d2s.append(self.body[i].weight[ch_mask[0, :, i, 1] == 1, ...].view(self.s_out_num[i], -1))
                else:
                    kernel_d2s.append([])
            else:
                if self.d_in_num[i] > 0 and self.d_out_num[i] > 0:
                    kernel_d2d.append(
                        self.body[i].weight[ch_mask[0, :, i, 0] == 1, ...][:, ch_mask[0, :, i - 1, 0] == 1, ...].view(
                            self.d_out_num[i], -1))
                else:
                    kernel_d2d.append([])
                if self.d_in_num[i] > 0 and self.s_out_num[i] > 0:
                    kernel_d2s.append(
                        self.body[i].weight[ch_mask[0, :, i, 1] == 1, ...][:, ch_mask[0, :, i - 1, 0] == 1, ...].view(
                            self.s_out_num[i], -1))
                else:
                    kernel_d2s.append([])
                if self.s_in_num[i] > 0:
                    kernel_s.append(torch.cat((
                        self.body[i].weight[ch_mask[0, :, i, 0] == 1, ...][:, ch_mask[0, :, i - 1, 1] == 1, ...],
                        self.body[i].weight[ch_mask[0, :, i, 1] == 1, ...][:, ch_mask[0, :, i - 1, 1] == 1, ...]),
                        0).view(self.d_out_num[i] + self.s_out_num[i], -1))
                else:
                    kernel_s.append([])

        # the last 1x1 conv
        ch_mask = ch_mask[0, ...].transpose(1, 0).contiguous().view(-1, 2)
        self.d_in_num.append(int(ch_mask[:, 0].sum(0)))
        self.s_in_num.append(int(ch_mask[:, 1].sum(0)))
        self.d_out_num.append(self.out_channels)
        self.s_out_num.append(0)

        kernel_d2d.append(self.collect.weight[:, ch_mask[..., 0] == 1, ...].squeeze())
        kernel_d2s.append([])
        kernel_s.append(self.collect.weight[:, ch_mask[..., 1] == 1, ...].squeeze())

        self.kernel_d2d = kernel_d2d
        self.kernel_d2s = kernel_d2s
        self.kernel_s = kernel_s
        self.bias = self.collect.bias

    def _generate_indices(self):
        A = torch.arange(3).to(self.spa_mask.device).view(-1, 1, 1)
        mask_indices = torch.nonzero(self.spa_mask.squeeze())

        # indices: dense to sparse (1x1)
        self.h_idx_1x1 = mask_indices[:, 0]
        self.w_idx_1x1 = mask_indices[:, 1]

        # indices: dense to sparse (3x3)
        mask_indices_repeat = mask_indices.unsqueeze(0).repeat([3, 1, 1]) + A

        self.h_idx_3x3 = mask_indices_repeat[..., 0].repeat(1, 3).view(-1)
        self.w_idx_3x3 = mask_indices_repeat[..., 1].repeat(3, 1).view(-1)

        # indices: sparse to sparse (3x3)
        indices = torch.arange(float(mask_indices.size(0))).view(1, -1).to(self.spa_mask.device) + 1
        self.spa_mask[0, 0, self.h_idx_1x1, self.w_idx_1x1] = indices

        self.idx_s2s = functional.pad(self.spa_mask, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9,
                                                                                                              -1).long()

    def _mask_select(self, x, k):
        if k == 1:
            return x[0, :, self.h_idx_1x1, self.w_idx_1x1]
        if k == 3:
            return functional.pad(x, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9 * x.size(1), -1)

    def _sparse_conv(self, fea_dense, fea_sparse, k, index):
        '''
        :param fea_dense: (B, C_d, H, W)
        :param fea_sparse: (C_s, N)
        :param k: kernel size
        :param index: layer index
        '''
        # dense input
        if self.d_in_num[index] > 0:
            if self.d_out_num[index] > 0:
                # dense to dense
                if k > 1:
                    fea_col = functional.unfold(fea_dense, k, stride=1, padding=(k - 1) // 2).squeeze(0)
                    fea_d2d = torch.mm(self.kernel_d2d[index].view(self.d_out_num[index], -1), fea_col)
                    fea_d2d = fea_d2d.view(1, self.d_out_num[index], fea_dense.size(2), fea_dense.size(3))
                else:
                    fea_col = fea_dense.view(self.d_in_num[index], -1)
                    fea_d2d = torch.mm(self.kernel_d2d[index].view(self.d_out_num[index], -1), fea_col)
                    fea_d2d = fea_d2d.view(1, self.d_out_num[index], fea_dense.size(2), fea_dense.size(3))

            if self.s_out_num[index] > 0:
                # dense to sparse
                fea_d2s = torch.mm(self.kernel_d2s[index], self._mask_select(fea_dense, k))

        # sparse input
        if self.s_in_num[index] > 0:
            # sparse to dense & sparse
            if k == 1:
                fea_s2ds = torch.mm(self.kernel_s[index], fea_sparse)
            else:
                fea_s2ds = torch.mm(self.kernel_s[index],
                                    functional.pad(fea_sparse,
                                                   [1, 0, 0, 0])[:, self.idx_s2s]
                                    .view(self.s_in_num[index] * k * k, -1))

        # fusion
        if self.d_out_num[index] > 0:
            if self.d_in_num[index] > 0:
                if self.s_in_num[index] > 0:
                    fea_d2d[0, :, self.h_idx_1x1, self.w_idx_1x1] += fea_s2ds[:self.d_out_num[index], :]
                    fea_d = fea_d2d
                else:
                    fea_d = fea_d2d
            else:
                fea_d = torch.zeros_like(self.spa_mask).repeat([1, self.d_out_num[index], 1, 1])
                fea_d[0, :, self.h_idx_1x1, self.w_idx_1x1] = fea_s2ds[:self.d_out_num[index], :]
        else:
            fea_d = None

        if self.s_out_num[index] > 0:
            if self.d_in_num[index] > 0:
                if self.s_in_num[index] > 0:
                    fea_s = fea_d2s + fea_s2ds[-self.s_out_num[index]:, :]
                else:
                    fea_s = fea_d2s
            else:
                fea_s = fea_s2ds[-self.s_out_num[index]:, :]
        else:
            fea_s = None

        # add bias (bias is only used in the last 1x1 conv in our SMB for simplicity)
        if index == 4:
            fea_d += self.bias.view(1, -1, 1, 1)

        return fea_d, fea_s

    def forward(self, x):
        '''
        :param x: [x[0], x[1]]
        x[0]: input feature (B, C ,H, W) ;
        x[1]: spatial mask (B, 1, H, W)
        '''
        if self.training:
            spa_mask = x[1]
            ch_mask = gumbel_softmax(self.ch_mask, 3, self.tau)

            out = []
            fea = x[0]
            for i in range(self.n_layers):
                if i == 0:
                    fea = self.body[i](fea)
                    fea = fea * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea * ch_mask[:, :, i:i + 1, :1]
                else:
                    fea_d = self.body[i](fea * ch_mask[:, :, i:i + 1, :1])
                    fea_s = self.body[i](fea * ch_mask[:, :, i:i + 1, 1:])
                    fea = fea_d * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea_d * ch_mask[:, :, i:i + 1, :1] + \
                          fea_s * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea_s * ch_mask[:, :, i:i + 1, :1] * spa_mask
                fea = self.relu(fea)
                out.append(fea)

            out = self.collect(torch.cat(out, 1))

            return out, ch_mask

        if not self.training:
            self.spa_mask = x[1]

            # generate indices
            self._generate_indices()

            # sparse conv
            fea_d = x[0]
            fea_s = None
            fea_dense = []
            fea_sparse = []
            for i in range(self.n_layers):
                fea_d, fea_s = self._sparse_conv(fea_d, fea_s, k=3, index=i)
                if fea_d is not None:
                    fea_dense.append(self.relu(fea_d))
                if fea_s is not None:
                    fea_sparse.append(self.relu(fea_s))

            # 1x1 conv
            fea_dense = torch.cat(fea_dense, 1)
            fea_sparse = torch.cat(fea_sparse, 0)
            out, _ = self._sparse_conv(fea_dense, fea_sparse, k=1, index=self.n_layers)

            return out


class SMM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SMM, self).__init__()

        # spatial mask
        self.spa_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels//4, in_channels//4, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels // 4, 2, 3, 2, 1, output_padding=1),
        )

        # body
        self.body = SMB(in_channels, out_channels, kernel_size, stride, padding, bias, n_layers=4)

        # CA layer
        self.ca = CALayer(out_channels)

        self.tau = 1

    def _update_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        if self.training:
            spa_mask = self.spa_mask(x)
            spa_mask = gumbel_softmax(spa_mask, 1, self.tau)

            out, ch_mask = self.body([x, spa_mask[:, 1:, ...]])
            out = self.ca(out) + x

            return out, spa_mask[:, 1:, ...], ch_mask

        if not self.training:
            spa_mask = self.spa_mask(x)
            spa_mask = (spa_mask[:, 1:, ...] > spa_mask[:, :1, ...]).float()

            out = self.body([x, spa_mask])
            out = self.ca(out) + x

            return out


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class SmsrModel(PreTrainedModel):
    config_class = SmsrConfig

    def __init__(self, args, conv=default_conv):
        super(SmsrModel, self).__init__(args)

        n_feats = args.n_feats
        kernel_size = 3
        self.scale = int(args.scale)

        # RGB mean for DIV2K
        rgb_mean = args.rgb_mean
        rgb_std = args.rgb_std
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size),
                        nn.ReLU(True),
                        conv(n_feats, n_feats, kernel_size)]

        # define body module
        modules_body = [SMM(n_feats, n_feats, kernel_size) \
                        for _ in range(5)]

        # define collect module
        self.collect = nn.Sequential(
            nn.Conv2d(64 * 5, 64, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1)
        )

        # define tail module
        modules_tail = [
            nn.Conv2d(n_feats, args.n_colors * self.scale * self.scale, 3, 1, 1),
            nn.PixelShuffle(self.scale),
        ]

        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x0 = self.sub_mean(x)
        x = self.head(x0)

        if self.training:
            sparsity = []
            out_fea = []
            fea = x
            for i in range(5):
                fea, _spa_mask, _ch_mask = self.body[i](fea)
                out_fea.append(fea)
                sparsity.append(
                    _spa_mask * _ch_mask[..., 1].view(1, -1, 1, 1) + torch.ones_like(_spa_mask) * _ch_mask[..., 0].view(
                        1, -1, 1, 1))
            out_fea = self.collect(torch.cat(out_fea, 1)) + x
            sparsity = torch.cat(sparsity, 0)

            x = self.tail(out_fea) + functional.interpolate(
                x0, scale_factor=self.scale, mode='bicubic', align_corners=False)
            x = self.add_mean(x)

            return [x, sparsity]

        if not self.training:
            out_fea = []
            fea = x
            for i in range(5):
                fea = self.body[i](fea)
                out_fea.append(fea)
            out_fea = self.collect(torch.cat(out_fea, 1)) + x

            x = self.tail(out_fea) + functional.interpolate(
                x0, scale_factor=self.scale, mode='bicubic', align_corners=False)
            x = self.add_mean(x)

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
