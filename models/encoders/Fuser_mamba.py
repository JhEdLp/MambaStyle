import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stylegan2.model import ModulatedConv2d, ToRGB
from models.encoders.vssm_arch import VSSBlock
from mamba_ssm import Mamba2

# Swish Function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

## ------------------------------------------------------------------------------------------------------------ ##

class LatentDirectionEncoder(nn.Module):
    def __init__(self):
        super(LatentDirectionEncoder, self).__init__()

        self.att_1 = Mamba2(512)
        self.SSD2D_1 = VSSBlock(hidden_dim=32, drop_path=0, norm_layer=nn.LayerNorm, attn_drop_rate=0, d_state=16, input_resolution=32) # SS2D(d_model=32)
        self.SSD2D_2 = VSSBlock(hidden_dim=512, drop_path=0, norm_layer=nn.LayerNorm, attn_drop_rate=0, d_state=16, input_resolution=64) # SS2D(d_model=512)

        self.conv1d_0 = nn.Conv1d(in_channels=18, out_channels=32, kernel_size=3, padding=1)
        self.conv1d_1 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)

        # Depthwise convolution: 32 input channels, 32 output channels, using groups=32 for depthwise
        self.depthwise_conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32)

        # Pointwise convolution: 32 input channels, 64 output channels
        self.pointwise_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)

        # Depthwise convolution: 64 input channels, 64 output channels
        self.depthwise_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)

        # Pointwise convolution: 64 input channels, 128 output channels
        self.pointwise_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)

        # ConvTranspose2d: Upsample from 32x32 to 64x64, keeping channel size at 128
        # self.conv_transpose = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)

        # Conv2d to refinement the Upsample feature map
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)

        # Final Conv2d layer to get the required output channels (512)
        self.final_conv = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

    def forward(self, latent_direction):

        latent_direction = self.conv1d_0(latent_direction)
        latent_direction = self.att_1(latent_direction)
        latent_direction = latent_direction.permute(0, 2, 1)
        latent_direction = self.conv1d_1(latent_direction).permute(1, 0, 2)
        # x = latent_direction.view(-1, 32, 32, 32) #  B x 32 x 32 x 32
        x = self.SSD2D_1(latent_direction) # .permute(0, 2, 3, 1).contiguous()
        x = x.view(1, 32, 32, -1).permute(0, 3, 1, 2).contiguous()

        # Apply depthwise and pointwise convolutions
        x = self.depthwise_conv1(x)  # B x 32 x 32 x 32
        x = self.pointwise_conv1(x)  # B x 64 x 32 x 32

        # Apply another round of depthwise and pointwise convolutions
        x = self.depthwise_conv2(x)  # B x 64 x 32 x 32
        x = self.pointwise_conv2(x)  # B x 128 x 32 x 32

        # Upsample to 64x64 using ConvTranspose2d
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)  # self.conv_transpose(x)  # B x 128 x 64 x 64
        x = self.conv1(x)
        x = self.conv2(x)
        # Final convolution to adjust output channels to 512
        x = self.final_conv(x).flatten(2).permute(2, 0, 1) # .permute(0, 2, 3, 1).contiguous()  # B x 64 x 64 x 512

        x = self.SSD2D_2(x).permute(1, 2, 0).view(1, 512, -1, 64).contiguous() # .permute(0, 3, 1, 2).contiguous()
        return x

class FeatureMapApproximator(nn.Module):
    def __init__(self, feature_dim=512):
        super(FeatureMapApproximator, self).__init__()
        # Combine original feature maps and latent direction to predict edited feature maps
        self.conv1 = nn.Conv2d(feature_dim * 2, 512, kernel_size=3, stride=1, padding=1, groups=feature_dim)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.SS2D_1 = VSSBlock(hidden_dim=512, drop_path=0, norm_layer=nn.LayerNorm, attn_drop_rate=0, d_state=16, input_resolution=64) # SS2D(d_model=512)

    def forward(self, feature_embedding, latent_embedding):
        # Concatenate along the channel dimension
        combined = torch.stack((feature_embedding, latent_embedding), dim=2) # Shape: (B, 512, 2, 64, 64)
        combined = combined.view(-1, 1024, 64, 64).contiguous()
        x = self.conv1(combined)
        x = self.conv2(x).flatten(2).permute(2, 0, 1) # .permute(0, 2, 3, 1)  # Output shape: (B, 64, 64, 512)
        x = self.SS2D_1(x) # .permute(0, 3, 1, 2).contiguous()
        return x

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
            approximated_feature_map = approximated_feature_map.permute(1, 2, 0).view(B, 512, -1, 64).contiguous()

            approximated_feature_map = self.modulate_conv_1(approximated_feature_map, latent_directions.mean(1, keepdim=True))
            approximated_feature_map = self.modulate_conv_2(approximated_feature_map, latent_directions.mean(1, keepdim=True))

            return approximated_feature_map

