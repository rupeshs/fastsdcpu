import torch
import torch.nn as nn

from .configuration_msrn import MsrnConfig
from ...modeling_utils import (
    default_conv,
    BamBlock,
    MeanShift,
    Upsampler,
    PreTrainedModel
)


class MsrBlock(nn.Module):
    def __init__(self, conv=default_conv, n_feats=64):
        super(MsrBlock, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class MsrBamBlock(nn.Module):
    def __init__(self, conv=default_conv, n_feats=64):
        super(MsrBamBlock, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.attention = BamBlock(n_feats)  # Add Balanced Attention

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.attention(self.confusion(input_3))  # Add Balanced Attention
        output += x
        return output


class MsrnModel(PreTrainedModel):
    config_class = MsrnConfig

    def __init__(self, args, conv=default_conv):
        super(MsrnModel, self).__init__(args)

        self.n_blocks = args.n_blocks
        n_feats = args.n_feats
        rgb_range = args.rgb_range
        kernel_size = 3
        scale = args.scale
        act = nn.ReLU(True)

        self.sub_mean = MeanShift(rgb_range, rgb_mean=args.rgb_mean, rgb_std=args.rgb_std)

        # define head module
        modules_head = [conv(3, n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()
        for i in range(self.n_blocks):
            if args.bam:
                modules_body.append(MsrBamBlock(n_feats=n_feats))
            else:
                modules_body.append(MsrBlock(n_feats=n_feats))

        # define tail module
        modules_tail = [
            nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats, 1, padding=0, stride=1),
            conv(n_feats, n_feats, kernel_size),
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]

        self.add_mean = MeanShift(rgb_range, sign=1, rgb_mean=args.rgb_mean, rgb_std=args.rgb_std)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)
        res = x

        MSRB_out = []
        for i in range(self.n_blocks):
            x = self.body[i](x)
            MSRB_out.append(x)
        MSRB_out.append(res)

        res = torch.cat(MSRB_out, 1)
        x = self.tail(res)
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
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError(f'While copying the parameter named {name}, '
                                           f'whose dimensions in the model are {own_state[name].size()} and '
                                           f'whose dimensions in the checkpoint are {param.size()}.')
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(f'unexpected key "{name}" in state_dict')

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError(f'missing keys in state_dict: "{missing}"')
