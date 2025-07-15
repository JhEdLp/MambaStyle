import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from timm.models.layers import DropPath, to_2tuple
from timm.models.layers import trunc_normal_, lecun_normal_
from functools import partial
import pdb

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class BackboneEncoderFirstStage(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderFirstStage, self).__init__()
        # print('Using BackboneEncoderFirstStage')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_3 = Sequential(BatchNorm2d(256),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(256 * 7 * 7, 512 * 8))
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(128 * 7 * 7, 512 * 4))
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(64 * 7 * 7, 512 * 4))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)

    def forward(self, x):
        x = self.input_layer(x)
        for l in self.modulelist[:3]:
            x = l(x)
        lc_part_4 = self.output_layer_5(x).view(-1, 4, 512)
        for l in self.modulelist[3:7]:
            x = l(x)
        lc_part_3 = self.output_layer_4(x).view(-1, 4, 512)
        for l in self.modulelist[7:21]:
            x = l(x)
        lc_part_2 = self.output_layer_3(x).view(-1, 8, 512)

        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        return x


class BackboneEncoderFirstStage_Mamba(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderFirstStage_Mamba, self).__init__()
        # print('Using BackboneEncoderFirstStage')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_3 = Sequential(torch.nn.BatchNorm1d(64),
                                         # torch.nn.Conv1d(64, 64, kernel_size=8, stride=8), # torch.nn.Conv1d(512, 64, kernel_size=1),
                                         torch.nn.AdaptiveAvgPool2d((1, 64)),
                                         Flatten(),
                                         Linear(64 * 64, 512 * 8))
        self.output_layer_4 = Sequential(torch.nn.BatchNorm1d(64),
                                         # torch.nn.Conv1d(64, 64, kernel_size=8, stride=8), # torch.nn.Conv1d(512, 64, kernel_size=1),
                                         torch.nn.AdaptiveAvgPool2d((1, 64)),
                                         Flatten(),
                                         Linear(64 * 64, 512 * 4))
        self.output_layer_5 = Sequential(torch.nn.BatchNorm1d(64),
                                         # torch.nn.Conv1d(64, 64, kernel_size=8, stride=8), # torch.nn.Conv1d(512, 64, kernel_size=1),
                                         torch.nn.AdaptiveAvgPool2d((1, 64)),
                                         Flatten(),
                                         Linear(64 * 64, 512 * 4))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)
        depth = 6

        self.patch_embed_5 = PatchEmbed(img_size=128, patch_size=16, stride=16, in_chans=64, embed_dim=512)
        self.patch_embed_4 = PatchEmbed(img_size=64, patch_size=8, stride=8, in_chans=128, embed_dim=512)
        self.patch_embed_3 = PatchEmbed(img_size=32, patch_size=4, stride=4, in_chans=256, embed_dim=512)

        dpr = [x.item() for x in torch.linspace(0, 0.05, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr

        self.layers_5 = nn.ModuleList([create_block(512, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, layer_idx=i,
                                                    if_bimamba=False, bimamba_type='v2', drop_path=inter_dpr[i], if_devide_out=True, init_layer_scale=None) for i in range(depth)])
        self.layers_4 = nn.ModuleList([create_block(512, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, layer_idx=i,
                                                    if_bimamba=False, bimamba_type='v2', drop_path=inter_dpr[i], if_devide_out=True, init_layer_scale=None) for i in range(depth)])
        self.layers_3 = nn.ModuleList([create_block(512, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, layer_idx=i,
                                                    if_bimamba=False, bimamba_type='v2', drop_path=inter_dpr[i], if_devide_out=True, init_layer_scale=None) for i in range(depth)])

        self.patch_embed_3.apply(segm_init_weights)
        self.layers_5.apply(partial(_init_weights, n_layer=depth, **{}))
        self.layers_4.apply(partial(_init_weights, n_layer=depth, **{}))
        self.layers_3.apply(partial(_init_weights, n_layer=depth, **{}))

    def forward(self, x):
        x = self.input_layer(x)

        for l in self.modulelist[:3]:
            x = l(x)

        x1 = self.patch_embed_5(x)
        residual = None
        for layer in self.layers_5:
            x1, residual = layer(x1, residual, inference_params=None)
        # lc_part_4 = self.output_layer_5(x1.permute(0, 2, 1)).view(-1, 4, 512)
        # lc_part_4 = self.output_layer_5(x1).view(-1, 4, 512)
        lc_part_4 = self.output_layer_5(x1[:, :, None, :]).view(-1, 4, 512)

        for l in self.modulelist[3:7]:
            x = l(x)

        x1 = self.patch_embed_4(x) + x1
        residual = None
        for layer in self.layers_4:
            x1, residual = layer(x1, residual, inference_params=None)
        # lc_part_3 = self.output_layer_4(x1.permute(0, 2, 1)).view(-1, 4, 512)
        # lc_part_3 = self.output_layer_4(x1).view(-1, 4, 512)
        lc_part_3 = self.output_layer_4(x1[:, :, None, :]).view(-1, 4, 512)

        for l in self.modulelist[7:21]:
            x = l(x)

        x1 = self.patch_embed_3(x) + x1
        residual = None
        for layer in self.layers_3:
            x1, residual = layer(x1, residual, inference_params=None)
        # lc_part_2 = self.output_layer_3(x1.permute(0, 2, 1)).view(-1, 8, 512)
        # lc_part_2 = self.output_layer_3(x1).view(-1, 8, 512)
        lc_part_2 = self.output_layer_3(x1[:, :, None, :]).view(-1, 8, 512)

        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        return x


class BackboneEncoderFirstStage_Transformer(Module):
    def __init__(self, num_layers, mode='ir', opts=None, depth=1):
        super(BackboneEncoderFirstStage_Transformer, self).__init__()
        # print('Using BackboneEncoderFirstStage')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_3 = Sequential(torch.nn.BatchNorm1d(64),
                                         # torch.nn.Conv1d(64, 64, kernel_size=8, stride=8), # torch.nn.Conv1d(512, 64, kernel_size=1),
                                         torch.nn.AdaptiveAvgPool2d((1, 64)),
                                         Flatten(),
                                         Linear(64 * 64, 512 * 8))
        self.output_layer_4 = Sequential(torch.nn.BatchNorm1d(64),
                                         # torch.nn.Conv1d(64, 64, kernel_size=8, stride=8), # torch.nn.Conv1d(512, 64, kernel_size=1),
                                         torch.nn.AdaptiveAvgPool2d((1, 64)),
                                         Flatten(),
                                         Linear(64 * 64, 512 * 4))
        self.output_layer_5 = Sequential(torch.nn.BatchNorm1d(64),
                                         # torch.nn.Conv1d(64, 64, kernel_size=8, stride=8), # torch.nn.Conv1d(512, 64, kernel_size=1),
                                         torch.nn.AdaptiveAvgPool2d((1, 64)),
                                         Flatten(),
                                         Linear(64 * 64, 512 * 4))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)

        self.patch_embed_c1 = PatchEmbed(img_size=64, patch_size=16, stride=16, in_chans=512, embed_dim=512)
        self.patch_embed_c2 = PatchEmbed(img_size=32, patch_size=8, stride=8, in_chans=512, embed_dim=512)
        self.patch_embed_c3 = PatchEmbed(img_size=16, patch_size=4, stride=4, in_chans=512, embed_dim=512)

        dpr = [x.item() for x in torch.linspace(0, 0.05, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr

        self.atte = nn.ModuleList([create_block(512, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, layer_idx=i,
                                                  if_bimamba=False, bimamba_type='v2', drop_path=inter_dpr[i], if_devide_out=True, init_layer_scale=None) for i in range(depth)])

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        for i, layer in enumerate(self.modulelist):
            x = layer(x)
            if i == 6:
                c1 = x  # torch.Size([8, 128, 64, 64])
            elif i == 20:
                c2 = x  # torch.Size([8, 256, 32, 32])
            if i == 23:
                c3 = x  # torch.Size([8, 512, 16, 16])

        c2 = self._upsample_add(c3, self.latlayer1(c2))
        c1 = self._upsample_add(c2, self.latlayer2(c1))

        c1 = self.patch_embed_c1(c1)
        c2 = self.patch_embed_c2(c2)
        c3 = self.patch_embed_c3(c3)

        x = c1 + c2 + c3

        residual = None
        for layer in self.atte:
            x, residual = layer(x, residual, inference_params=None)

        return x


class Mamba(Module):
    def __init__(self, num_layers, mode='ir', opts=None, depth=24):
        super(Mamba, self).__init__()
        # print('Using BackboneEncoderFirstStage')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'

        drop_path_rate = 0.05
        drop_rate = 0.0
        rms_norm = True
        embed_dim = 512
        norm_epsilon = 1e-5

        self.embed_dim = 512

        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList([create_block(embed_dim, ssm_cfg=None, norm_epsilon=norm_epsilon, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, layer_idx=i,
                                                  if_bimamba=False, bimamba_type='v2', drop_path=inter_dpr[i], if_devide_out=True, init_layer_scale=None) for i in range(depth)])

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon)

        self.patch_embed.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(partial(_init_weights, n_layer=depth, **{}))

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.pool1 = nn.AdaptiveAvgPool2d((8, 8))
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        residual = None

        for layer in self.layers:
            x, residual = layer(x, residual, inference_params=None)

        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn

        x = fused_add_norm_fn(self.drop_path(x), self.norm_f.weight, self.norm_f.bias, eps=self.norm_f.eps, residual=residual, prenorm=False,
                              residual_in_fp32=True)

        B, _, _ = x.shape

        x = x.view(B, 16, 16, self.embed_dim)
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.flatten(2)
        x = x.permute(0, 2, 1)
        return x


class BackboneEncoderRefineStage(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderRefineStage, self).__init__()
        # print('Using BackboneEncoderRefineStage')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(6, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_3 = Sequential(BatchNorm2d(256),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(256 * 7 * 7, 512 * 9))
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(128 * 7 * 7, 512 * 5))
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(64 * 7 * 7, 512 * 4))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)

    def forward(self, x, first_stage_output_image):
        x = torch.cat((x, first_stage_output_image), dim=1)
        x = self.input_layer(x)
        for l in self.modulelist[:3]:
            x = l(x)
        lc_part_4 = self.output_layer_5(x).view(-1, 4, 512)
        for l in self.modulelist[3:7]:
            x = l(x)
        lc_part_3 = self.output_layer_4(x).view(-1, 5, 512)
        for l in self.modulelist[7:21]:
            x = l(x)
        lc_part_2 = self.output_layer_3(x).view(-1, 9, 512)

        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=16, stride=16, in_chans=3, embed_dim=512, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)
