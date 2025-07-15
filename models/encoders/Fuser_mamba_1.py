import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stylegan2.model import ModulatedConv2d, ToRGB
from models.encoders.vssm_arch import VSSBlock
from mamba_ssm import Mamba2
import pdb

# Swish Function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

## ------------------------------------------------------------------------------------------------------------ ##

class LatentDirectionEncoder(nn.Module):
    def __init__(self):
        super(LatentDirectionEncoder, self).__init__()

        self.att_1 = Mamba2(512)
        self.SSD2D_1 = VSSBlock(hidden_dim=32, drop_path=0, norm_layer=nn.LayerNorm, 
                              attn_drop_rate=0, d_state=16, input_resolution=32)
        self.SSD2D_2 = VSSBlock(hidden_dim=512, drop_path=0, norm_layer=nn.LayerNorm, 
                              attn_drop_rate=0, d_state=16, input_resolution=64)

        self.conv1d_0 = nn.Conv1d(18, 32, kernel_size=3, padding=1)
        self.conv1d_1 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)

        # Depthwise/Pointwise Convs
        self.depthwise_conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pointwise_conv1 = nn.Conv2d(32, 64, kernel_size=1)
        self.norm1 = nn.InstanceNorm2d(64)
        
        self.depthwise_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.pointwise_conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.norm2 = nn.InstanceNorm2d(128)

        # Upsampling
        self.upsample_conv = nn.Conv2d(128, 128*4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.blur = nn.AvgPool2d(3, stride=1, padding=1)

        # Final Convs
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=1)
        self.final_conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # GELU activations with proper initialization
        self.act = nn.GELU()

        # Normalization layers for GELU compatibility
        self.norm1 = nn.LayerNorm(64)  # LayerNorm works better with GELU
        self.norm2 = nn.LayerNorm(128)

    def forward(self, latent_direction):
        # Initial processing
        x = self.conv1d_0(latent_direction)
        x = self.att_1(x)
        x = x.permute(0, 2, 1)
        x = self.conv1d_1(x).permute(1, 0, 2)
        
        # Reshape with dynamic batch size
        B = x.size(1)
        x = self.SSD2D_1(x)
        x = x.view(B, 32, 32, -1).permute(0, 3, 1, 2).contiguous()

        # Conv processing
        x = self.pointwise_conv1(x)
        x = x.permute(0, 2, 3, 1)  # Channel last for LayerNorm
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # Back to channel first
        x = self.act(x)

        # Repeat for second block
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)

        # Upsample with pixel shuffle
        x = self.upsample_conv(x)
        x = self.pixel_shuffle(x)
        x = self.blur(x)  # Smooth artifacts

        # Final layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_conv(x).flatten(2).permute(2, 0, 1)
        
        x = self.SSD2D_2(x).permute(1, 2, 0).view(B, 512, -1, 64).contiguous()
        return x

class FeatureMapApproximator(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        
        # Learnable fusion weights with channel attention
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(feature_dim*2, feature_dim//16, 1),
            nn.GELU(),
            nn.Conv2d(feature_dim//16, feature_dim*2, 1),
            nn.Sigmoid()
        )

        # Artifact-aware fusion blocks
        self.fusion_blocks = nn.Sequential(
            nn.Conv2d(feature_dim*2, feature_dim, 3, padding=1, bias=False),
            nn.LayerNorm([feature_dim, 64, 64]),  # Spatial-channel LN
            nn.GELU(),
            
            # Dilated convolution for multi-scale context
            nn.Conv2d(feature_dim, feature_dim, 3, padding=2, dilation=2, groups=feature_dim),
            nn.Conv2d(feature_dim, feature_dim, 1),  # Pointwise
            nn.GELU(),
            
            # Channel shuffle for better mixing
            ChannelShuffle(groups=8),
            
            # Final projection
            nn.Conv2d(feature_dim, feature_dim, 1)
        )

        # Spatial attention refinement
        self.SS2D_1 = VSSBlock(hidden_dim=feature_dim, 
                              drop_path=0, 
                              norm_layer=nn.LayerNorm,
                              attn_drop_rate=0,
                              d_state=16,
                              input_resolution=64)

        # Post-attention anti-aliasing
        self.blur_pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, feature_embedding, latent_embedding):
        # Adaptive feature fusion
        concat_features = torch.cat([feature_embedding, latent_embedding], dim=1)
        
        # Learnable fusion gating
        fusion_weights = self.fusion_gate(concat_features)
        weighted_features = concat_features * fusion_weights
        
        # Progressive fusion with residual
        fused = self.fusion_blocks(weighted_features) + feature_embedding
        
        # Attention refinement
        B, C, H, W = fused.shape
        attn_input = fused.flatten(2).permute(2, 0, 1) 
        attn_output = self.SS2D_1(attn_input).permute(1, 2, 0).view(B, C, -1, W).contiguous()
        
        # Anti-aliasing and residual
        return self.blur_pool(attn_output) + fused

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.groups, C//self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(B, C, H, W)

class Fuser_Model(nn.Module):
    def __init__(self):
        super(Fuser_Model, self).__init__()
        self.latent_encoder = LatentDirectionEncoder()
        self.approximator = FeatureMapApproximator()
        self.modulate_conv_1 = ModulatedConv2d(in_channel=512, out_channel=512, kernel_size=3, style_dim=512, demodulate=True, upsample=False, downsample=False)
        self.modulate_conv_2 = ModulatedConv2d(in_channel=512, out_channel=512, kernel_size=3, style_dim=512, demodulate=True, upsample=False, downsample=False)

    def forward(self, feature_map, latent_directions):
            B, _, _, _ = feature_map.size()
            # Encode feature map and latent directions
            latent_embedding = self.latent_encoder(latent_directions)
            latent_embedding = latent_embedding.repeat(B, 1, 1, 1)
            latent_directions = latent_directions.repeat(B, 1, 1)
            approximated_feature_map = self.approximator(feature_map, latent_embedding)

            approximated_feature_map = self.modulate_conv_1(approximated_feature_map, latent_directions.mean(1, keepdim=True))
            approximated_feature_map = self.modulate_conv_2(approximated_feature_map, latent_directions.mean(1, keepdim=True))

            return approximated_feature_map

