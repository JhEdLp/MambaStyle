import math

import numpy as np
import torch
from torch.nn import functional as F, init
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE, PatchEmbed, PatchEmbed_1, _init_weights, segm_init_weights, Unflatten, PatchMerging
from torch_utils.ops import bias_act
from criteria.parsing_loss.model_utils import *
import math
from models.encoders.vssm_arch import VSSBlock



class Backbone_Mamba(Module):
    def __init__(self, num_layers, mode='ir', opts=None, depth=1):
        super(Backbone_Mamba, self).__init__()
        # print('Using BackboneEncoderFirstStage')
        assert num_layers in [15, 34, 50, 100, 152], 'num_layers should be 34, 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        self.num_layers = num_layers
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        if num_layers == 34:
            self.stop1, self.stop2, self.stop3 = 3, 8, 10
        elif num_layers == 50:
            self.stop1, self.stop2, self.stop3 = 6, 20, 23
        elif num_layers == 15:
            self.stop1, self.stop2, self.stop3 = 3, 5, 7

        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_1 = Sequential(torch.nn.BatchNorm2d(512),
                                         Unflatten(),
                                         torch.nn.AdaptiveAvgPool3d((64, 9, 9)),
                                         Flatten(),
                                         Linear(64 * 9 * 9, 512 * 4))

        self.output_layer_2 = Sequential(torch.nn.BatchNorm2d(512),
                                         Unflatten(),
                                         torch.nn.AdaptiveAvgPool3d((64, 6, 6)),
                                         Flatten(),
                                         Linear(64 * 6 * 6, 512 * 5))

        self.output_layer_3 = Sequential(torch.nn.BatchNorm2d(512),
                                         Unflatten(),
                                         torch.nn.AdaptiveAvgPool3d((128, 5, 5)),
                                         Flatten(),
                                         Linear(128 * 5 * 5, 512 * 9))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)

        dpr = [x.item() for x in torch.linspace(0, 0.05, 24)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr

        self.atte_1 = nn.ModuleList([create_block(512, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, layer_idx=i,
                                                  if_bimamba=False, bimamba_type=None, drop_path=inter_dpr[i], if_devide_out=True, init_layer_scale=None) for i in range(depth)])
        self.atte_2 = nn.ModuleList([create_block(512, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, layer_idx=i,
                                                  if_bimamba=False, bimamba_type=None, drop_path=inter_dpr[i], if_devide_out=True, init_layer_scale=None) for i in range(depth)])
        self.atte_3 = nn.ModuleList([create_block(512, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, layer_idx=i,
                                                  if_bimamba=False, bimamba_type=None, drop_path=inter_dpr[i], if_devide_out=True, init_layer_scale=None) for i in range(depth)])

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.adapt_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):

        B, _, _, _ = x.size()
        x = self.input_layer(x)

        for i, layer in enumerate(self.modulelist):
            x = layer(x)
            if i == self.stop1:
                c1 = x  # torch.Size([8, 128, 64, 64])
            elif i == self.stop2:
                c2 = x  # torch.Size([8, 256, 32, 32])
            if i == self.stop3:
                c3 = x  # torch.Size([8, 512, 16, 16])

        c2 = self._upsample_add(c3, self.latlayer1(c2))
        c1 = self._upsample_add(c2, self.latlayer2(c1))

        c1 = c1.flatten(2).permute(0, 2, 1)
        c2 = c2.flatten(2).permute(0, 2, 1)
        c3 = c3.flatten(2).permute(0, 2, 1)
        # aux.flatten(2).permute(0, 2, 1).permute(0, 2, 1).reshape(B, 128, 64, 64)

        residual = None
        for layer in self.atte_1:
            c1, residual = layer(c1, residual, inference_params=None)

        residual = None
        for layer in self.atte_2:
            c2, residual = layer(c2, residual, inference_params=None)

        residual = None
        for layer in self.atte_3:
            c3, residual = layer(c3, residual, inference_params=None)

        c1 = c1.permute(0, 2, 1).reshape(B, 512, 64, 64).contiguous()
        c2 = c2.permute(0, 2, 1).reshape(B, 512, 32, 32).contiguous()
        c3 = c3.permute(0, 2, 1).reshape(B, 512, 16, 16).contiguous()

        C = self.adapt_1(c1)

        c1 = self.output_layer_1(c1).view(-1, 4, 512)
        c2 = self.output_layer_2(c2).view(-1, 5, 512)
        c3 = self.output_layer_3(c3).view(-1, 9, 512)

        x = torch.cat([c1, c2, c3], dim=1)

        return x, C

class Backbone_Mamba_ffhq(Module):
    def __init__(self, num_layers, mode='ir', opts=None, depth=1):
        super(Backbone_Mamba_ffhq, self).__init__()
        # print('Using BackboneEncoderFirstStage')
        assert num_layers in [15, 34, 50, 100, 152], 'num_layers should be 34, 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        self.num_layers = num_layers
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        if num_layers == 34:
            self.stop1, self.stop2, self.stop3 = 3, 8, 10
        elif num_layers == 50:
            self.stop1, self.stop2, self.stop3 = 6, 20, 23
        elif num_layers == 15:
            self.stop1, self.stop2, self.stop3 = 3, 5, 7

        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_1 = Sequential(torch.nn.BatchNorm2d(512),
                                         Unflatten(),
                                         torch.nn.AdaptiveAvgPool3d((64, 9, 9)),
                                         Flatten(),
                                         Linear(64 * 9 * 9, 512 * 4))

        self.output_layer_2 = Sequential(torch.nn.BatchNorm2d(512),
                                         Unflatten(),
                                         torch.nn.AdaptiveAvgPool3d((64, 6, 6)),
                                         Flatten(),
                                         Linear(64 * 6 * 6, 512 * 5))

        self.output_layer_3 = Sequential(torch.nn.BatchNorm2d(512),
                                         Unflatten(),
                                         torch.nn.AdaptiveAvgPool3d((128, 5, 5)),
                                         Flatten(),
                                         Linear(128 * 5 * 5, 512 * 9))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)


        self.SS2D_1 = VSSBlock(hidden_dim=512, drop_path=0, norm_layer=nn.LayerNorm, attn_drop_rate=0, d_state=16, input_resolution=64) # SS2D(d_model=512)
        self.SS2D_2 = VSSBlock(hidden_dim=512, drop_path=0, norm_layer=nn.LayerNorm, attn_drop_rate=0, d_state=16, input_resolution=64) # SS2D(d_model=512)
        self.SS2D_3 = VSSBlock(hidden_dim=512, drop_path=0, norm_layer=nn.LayerNorm, attn_drop_rate=0, d_state=16, input_resolution=64) # SS2D(d_model=512)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.adapt_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.proj = PatchEmbed(img_size=256, patch_size=8, in_chans=3)
        self.new_ps = nn.Conv2d(512 , 512 , (1,1))
        self.averagepooling = nn.AdaptiveAvgPool2d(18)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):

        B, _, _, _ = x.size()
        projection = self.proj(x)
        content_pool = self.averagepooling(projection)
        pos_c = self.new_ps(content_pool)
        pos_embed_c = F.interpolate(pos_c, mode='bilinear', size=(32,32))

        x = self.input_layer(x)

        for i, layer in enumerate(self.modulelist):
            x = layer(x)
            if i == self.stop1:
                c1 = x  # torch.Size([8, 128, 64, 64])
            elif i == self.stop2:
                c2 = x  # torch.Size([8, 256, 32, 32])
            if i == self.stop3:
                c3 = x  # torch.Size([8, 512, 16, 16])

        c2 = self._upsample_add(c3, self.latlayer1(c2)) + pos_embed_c
        c1 = self._upsample_add(c2, self.latlayer2(c1))

        c1 = c1.flatten(2).permute(2, 0, 1)
        c2 = c2.flatten(2).permute(2, 0, 1)
        c3 = c3.flatten(2).permute(2, 0, 1)

        c1 = self.SS2D_1(c1).permute(1, 2, 0).view(B, 512, -1, 64).contiguous()
        c2 = self.SS2D_2(c2).permute(1, 2, 0).view(B, 512, -1, 32).contiguous()
        c3 = self.SS2D_3(c3).permute(1, 2, 0).view(B, 512, -1, 16).contiguous()

        C = self.adapt_1(c1)

        c1 = self.output_layer_1(c1).view(-1, 4, 512)
        c2 = self.output_layer_2(c2).view(-1, 5, 512)
        c3 = self.output_layer_3(c3).view(-1, 9, 512)

        x = torch.cat([c1, c2, c3], dim=1)

        return x, C

class Backbone_Mamba_cars(Module):
    def __init__(self, num_layers, mode='ir', opts=None, depth=1):
        super(Backbone_Mamba_cars, self).__init__()
        # print('Using BackboneEncoderFirstStage')
        assert num_layers in [15, 34, 50, 100, 152], 'num_layers should be 34, 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        self.num_layers = num_layers
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        if num_layers == 34:
            self.stop1, self.stop2, self.stop3 = 3, 8, 10
        elif num_layers == 50:
            self.stop1, self.stop2, self.stop3 = 6, 20, 23
        elif num_layers == 15:
            self.stop1, self.stop2, self.stop3 = 3, 5, 7

        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_1 = Sequential(torch.nn.BatchNorm2d(512),
                                         Unflatten(),
                                         torch.nn.AdaptiveAvgPool3d((64, 9, 9)),
                                         Flatten(),
                                         Linear(64 * 9 * 9, 512 * 4))

        self.output_layer_2 = Sequential(torch.nn.BatchNorm2d(512),
                                         Unflatten(),
                                         torch.nn.AdaptiveAvgPool3d((64, 6, 6)),
                                         Flatten(),
                                         Linear(64 * 6 * 6, 512 * 4))

        self.output_layer_3 = Sequential(torch.nn.BatchNorm2d(512),
                                         Unflatten(),
                                         torch.nn.AdaptiveAvgPool3d((128, 5, 5)),
                                         Flatten(),
                                         Linear(128 * 5 * 5, 512 * 8))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)


        self.SS2D_1 = VSSBlock(hidden_dim=512, drop_path=0, norm_layer=nn.LayerNorm, attn_drop_rate=0, d_state=16, input_resolution=64) # SS2D(d_model=512)
        self.SS2D_2 = VSSBlock(hidden_dim=512, drop_path=0, norm_layer=nn.LayerNorm, attn_drop_rate=0, d_state=16, input_resolution=64) # SS2D(d_model=512)
        self.SS2D_3 = VSSBlock(hidden_dim=512, drop_path=0, norm_layer=nn.LayerNorm, attn_drop_rate=0, d_state=16, input_resolution=64) # SS2D(d_model=512)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.adapt_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.proj = PatchEmbed(img_size=256, patch_size=8, in_chans=3)
        self.new_ps = nn.Conv2d(512 , 512 , (1,1))
        self.averagepooling = nn.AdaptiveAvgPool2d(18)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):

        B, _, _, _ = x.size()
        projection = self.proj(x)
        content_pool = self.averagepooling(projection)
        pos_c = self.new_ps(content_pool)
        pos_embed_c = F.interpolate(pos_c, mode='bilinear', size=(32,32))

        x = self.input_layer(x)

        for i, layer in enumerate(self.modulelist):
            x = layer(x)
            if i == self.stop1:
                c1 = x  # torch.Size([8, 128, 64, 64])
            elif i == self.stop2:
                c2 = x  # torch.Size([8, 256, 32, 32])
            if i == self.stop3:
                c3 = x  # torch.Size([8, 512, 16, 16])

        c2 = self._upsample_add(c3, self.latlayer1(c2)) + pos_embed_c
        c1 = self._upsample_add(c2, self.latlayer2(c1))

        c1 = c1.flatten(2).permute(2, 0, 1)
        c2 = c2.flatten(2).permute(2, 0, 1)
        c3 = c3.flatten(2).permute(2, 0, 1)

        c1 = self.SS2D_1(c1).permute(1, 2, 0).view(B, 512, -1, 64).contiguous()
        c2 = self.SS2D_2(c2).permute(1, 2, 0).view(B, 512, -1, 32).contiguous()
        c3 = self.SS2D_3(c3).permute(1, 2, 0).view(B, 512, -1, 16).contiguous()

        C = self.adapt_1(c1)

        c1 = self.output_layer_1(c1).view(-1, 4, 512)
        c2 = self.output_layer_2(c2).view(-1, 4, 512)
        c3 = self.output_layer_3(c3).view(-1, 8, 512)

        x = torch.cat([c1, c2, c3], dim=1)

        return x, C

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)