import torch
from torch import nn
from models.encoders import backbone_encoders
from models.stylegan2.model import Generator, Discriminator
from configs.paths_config import model_paths
from models.encoders import mamba_encoders
# from models.stylegan3.stylegan3 import Generator
from models.encoders.helpers import ResBlk
from models.encoders.Fuser_mamba import Fuser_Model
import random
import numpy as np
from editings.latent_editor import LatentEditor
import pdb


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if (k[:len(name)] == name) and (k[len(name)] != '_')}
    return d_filt


class Mamba_Inv(nn.Module):

    def __init__(self, opts):
        super(Mamba_Inv, self).__init__()
        self.optimizer_ckpt = None
        self.global_step = None
        self.set_opts(opts)

        self.encoder = mamba_encoders.Backbone_Mamba_cars(34, 'ir_se', self.opts, depth=1) if "car" in opts.dataset_type else mamba_encoders.Backbone_Mamba_ffhq(34, 'ir_se', self.opts, depth=1)
        self.decoder = Generator(512 if "car" in opts.dataset_type else 1024, 512, 8)
        self.fuser = Fuser_Model()
        # self.discriminator =  Discriminator(1024, channel_multiplier=2)

        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()


        self.direction = {}
        aux = np.load('./editing_directions/Eyeglasses_boundary.npy')
        self.direction['glasses'] = torch.FloatTensor(aux).reshape(1, 18, 512).cuda()
        aux = np.load('./editing_directions/Heavy_Makeup_boundary.npy')
        self.direction['makeup'] = torch.FloatTensor(aux).reshape(1, 18, 512).cuda()
        aux = np.load('./editing_directions/Smiling_boundary.npy')
        self.direction['smile'] = torch.FloatTensor(aux).reshape(1, 18, 512).cuda()
        aux = torch.load('./editing_directions/age.pt')
        self.direction['age'] = torch.FloatTensor(aux).repeat(1, 18, 1).cuda()

        self.latent_editor = LatentEditor("car" if "car" in opts.dataset_type else "human_faces")


    def load_weights(self):
        if (self.opts.checkpoint_path is not None) and (not self.opts.is_training):
            print(f'Preparing the Model for Inference....', flush=True)
            print('Loading Model from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.fuser.load_state_dict(get_keys(ckpt, 'fuser'), strict=True)
            self.__load_latent_avg(ckpt)
        elif (self.opts.checkpoint_path is not None) and self.opts.is_training:
            print(f'Training the model from previous checkpoint...', flush=True)
            print('Loading previous encoders and decoder from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.fuser.load_state_dict(get_keys(ckpt, 'fuser'), strict=True)
            self.__load_latent_avg(ckpt)

            # ckpt = torch.load(self.opts.stylegan_weights)
            # self.discriminator.load_state_dict(get_keys(ckpt, 'discriminator'), strict=True)
        elif (self.opts.checkpoint_path is None) and self.opts.is_training:
            print(f'Train: The Model, encoder is to be trained.', flush=True)
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            # if input to encoder is not an RGB image, do not load the input layer weights
            if self.opts.label_nc != 0:
                encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
            # self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=True)
            # self.discriminator.load_state_dict(ckpt['d'], strict=True)
            if self.opts.learn_in_w:
                self.__load_latent_avg(ckpt, repeat=1)
            else:
                if "car" in self.opts.dataset_type:
                    self.__load_latent_avg(ckpt, repeat=1)
                else:
                    self.__load_latent_avg(ckpt, repeat=18)

        if 'global_step' in ckpt.keys():
            self.global_step = int(ckpt['global_step'])

        if 'optimizer' in ckpt.keys():
            self.optimizer_ckpt = ckpt['optimizer']

    def forward(self, x, resize=True, input_code=False, randomize_noise=True, return_latents=False, strength=0.0, direction_name="Inversion"):

        B, _, _, _ = x.shape
        if input_code:
            codes = x
        else:
            codes, C = self.encoder(x)
            if self.opts.start_from_latent_avg:
                if self.opts.learn_in_w:
                    codes = codes + self.latent_avg.repeat(B, 1)
                else:
                    codes = codes + self.latent_avg.repeat(B, 1, 1)

        input_is_latent = not input_code
        # direction = 0.0 * self.direction['glasses'] # [:,:16, :] #5.repeat(B, 1, 1) # 12, -10 power of glasses and smile
        if direction_name == "Inversion":
            if "car" in self.opts.dataset_type:
                direction = torch.zeros((1, 16, 512), device=self.opts.device)
            else :
                direction = torch.zeros((1, 18, 512), device=self.opts.device)

        else:
            we = self.get_edited_latent(codes, direction_name, [strength])[0]
            direction = we - codes
            direction = direction.mean(0, keepdim=True)
            codes = we
            # direction = self.direction['glasses'] * strength
            # codes += direction

        F0 = self.fuser(C, direction)
        output, result_latent = self.decoder([codes], input_is_latent=input_is_latent, randomize_noise=randomize_noise, return_latents=return_latents, feats=F0)

        if resize:
            output = self.face_pool(output)

        if return_latents:
            return output, result_latent
        else:
            return output

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    def get_edited_latent(self, original_latent, editing_name, editing_degrees, original_image=None):

        if self.latent_editor.domain == "human_faces":
            if editing_name in self.latent_editor.interfacegan_directions:
                edited_latents = self.latent_editor.get_interface_gan_edits(original_latent, editing_degrees, editing_name)

            elif editing_name in self.latent_editor.styleclip_directions:
                edited_latents = self.latent_editor.get_styleclip_mapper_edits(original_latent, editing_degrees, editing_name)

            elif editing_name in self.latent_editor.ganspace_directions:
                edited_latents = self.latent_editor.get_ganspace_edits(original_latent, editing_degrees, editing_name)

            elif editing_name in self.latent_editor.fs_directions.keys():
                edited_latents = self.latent_editor.get_fs_edits(original_latent, editing_degrees, editing_name)

            else:
                raise ValueError(f'Edit name {editing_name} is not available')
        else:
            if editing_name in self.latent_editor.ganspace_directions:
                edited_latents = self.latent_editor.get_ganspace_edits(original_latent, editing_degrees, editing_name)
            else:
                raise ValueError(f'Edit name {editing_name} is not available')
        return edited_latents

