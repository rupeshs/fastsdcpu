import torch
import torch.nn as nn
import torch.nn.functional as functional

from .configuration_jiif import JiifConfig
from .. import EdsrModel, EdsrConfig
from ...modeling_utils import PreTrainedModel


def make_edsr_baseline(scale=2, n_feats=16, no_upsampling=True, n_colors=1):
    return EdsrModel(EdsrConfig(
        scale=scale,
        n_feats=n_feats,
        n_colors=n_colors,
        no_upsampling=no_upsampling,
    ))


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    ranged in [-1, 1]
    e.g.
        shape = [2] get (-0.5, 0.5)
        shape = [3] get (-0.67, 0, 0.67)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)  # [H, W, 2]
    if flatten:
        ret = ret.view(-1, ret.shape[-1])  # [H*W, 2]
    return ret


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class JiifModel(PreTrainedModel):
    config_class = JiifConfig

    def __init__(self, args):
        super(JiifModel, self).__init__(args)

        self.args = args
        self.scale = args.scale
        self.feat_dim = args.feat_dim
        self.guide_dim = args.guide_dim
        self.mlp_dim = args.mlp_dim

        self.image_encoder = make_edsr_baseline(scale=self.scale, n_feats=self.guide_dim, n_colors=3)
        self.depth_encoder = make_edsr_baseline(scale=self.scale, n_feats=self.feat_dim, n_colors=1)

        imnet_in_dim = self.feat_dim + self.guide_dim * 2 + 2

        self.imnet = MLP(imnet_in_dim, out_dim=2, hidden_list=self.mlp_dim)

    def query(self, feat, coord, hr_guide, lr_guide, image):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)

        q_guide_hr = functional.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                     :, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        k = 0
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry
                k += 1

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = functional.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest',
                                                align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [B, N, c]
                q_coord = functional.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest',
                                                 align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                q_guide_lr = functional.grid_sample(lr_guide, coord_.flip(-1).unsqueeze(1), mode='nearest',
                                                    align_corners=False)[
                             :, :, 0, :].permute(0, 2, 1)  # [B, N, C]
                q_guide = torch.cat([q_guide_hr, q_guide_hr - q_guide_lr], dim=-1)

                inp = torch.cat([q_feat, q_guide, rel_coord], dim=-1)

                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = functional.softmax(preds[:, :, 1, :], dim=-1)

        ret = (preds[:, :, 0, :] * weight).sum(-1, keepdim=True)

        return ret

    def forward(self, data):
        image, depth, coord, res, lr_image = data['image'], data['lr'], data['hr_coord'], data['lr_pixel'], data[
            'lr_image']

        hr_guide = self.image_encoder(image)
        lr_guide = self.image_encoder(lr_image)

        feat = self.depth_encoder(depth)

        if self.training or not self.args.batched_eval:
            res = res + self.query(feat, coord, hr_guide, lr_guide, data['hr_depth'].repeat(1, 3, 1, 1))

        # batched evaluation to avoid OOM
        else:
            N = coord.shape[1]  # coord ~ [B, N, 2]
            n = 30720
            tmp = []
            for start in range(0, N, n):
                end = min(N, start + n)
                ans = self.query(feat, coord[:, start:end], hr_guide, lr_guide,
                                 data['hr_depth'].repeat(1, 3, 1, 1))  # [B, N, 1]
                tmp.append(ans)
            res = res + torch.cat(tmp, dim=1)

        return res

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
