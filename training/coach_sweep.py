import os
import matplotlib.pyplot as plt

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from criteria.parsing_loss import parse_loss
from utils import common, train_utils
from criteria import id_loss, w_norm
from configs import data_configs
from datasets_1.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.inv_mamba import Mamba_Inv
from training.ranger import Ranger
from torch.optim.lr_scheduler import MultiStepLR
import random
import numpy as np
import pdb
from editings.latent_editor import LatentEditor
from utils.editing_utils import get_stylespace_from_w


FACE_DIRECTIONS = {
    "age": [-7, -5, -3, 3, 5, 7, 10],
    "fs_makeup": [5, 8, 10, 12],
    "afro": [0.1, 0.15, 0.18  ],
    "angry": [0.1, 0.125, 0.15],
    "purple_hair": [0.1, 0.125, 0.15],
    "fs_glasses": [10, 12.5, 15, 17.5, 20, 22],
    "face_roundness": [-13, -7, 7, 13],
    "rotation": [-7.0, -5.0, -3.0, 3.0, 5.0, 7.0],
    "bobcut": [0.1, 0.12, 0.18],
    "bowlcut": [0.08, 0.11, 0.15, 0.18],
    "mohawk": [0.08, 0.11, 0.15],
    "fs_smiling": [-6, -3, 3, 6, 9]
}


def get_random_edit():
    direction = np.random.choice(list(FACE_DIRECTIONS.keys()))
    strenght = np.random.choice(FACE_DIRECTIONS[direction])
    return direction, strenght


class Coach:
    def __init__(self, opts):
        self.wandb_logger = True

        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
        self.opts.device = self.device

        # Initialize network
        self.net = Mamba_Inv(self.opts).to(self.device)

        # Initialize loss
        if self.opts.lpips_lambda_inv > 0 or self.opts.lpips_lambda_edit >0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda_inv > 0 or self.opts.lpips_lambda_edit >0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        if self.opts.parse_lambda_inv > 0 or self.opts.lpips_lambda_edit >0:
            self.parse_loss = parse_loss.ParseLoss().to(self.device).eval()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=400000, gamma=self.opts.gamma_scheduler)

        if self.net.global_step is not None:
            # self.global_step = self.net.global_step
            pass

        if self.net.optimizer_ckpt is not None:
            # self.optimizer.load_state_dict(self.net.optimizer_ckpt)
            pass


        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

        self.latent_editor = LatentEditor()


    def train_jhon(self):

        self.net.encoder.train()
        self.net.fuser.train()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

        if not self.opts.train_decoder:
            self.net.decoder.eval()
        else:
            self.net.decoder.train()

        while self.global_step < self.opts.max_steps:

            with torch.no_grad():
                # Generate latent codes in W+ space using pre-trained StyleGAN2 mapping network
                latent_noise = torch.randn([self.opts.batch_size, 18, 512]).cuda()
                w = self.net.decoder.style(latent_noise)# Map z to W space using the mapping network
                # w = w.unsqueeze(1).repeat(1, 18, 1)  # Expand w to W+ (18 latent vectors for 1024x1024 resolution)
                w = w + self.net.latent_avg[None, ...]

                # Adjust latent code with random direction

                d, strength = get_random_edit()  # Get random edit direction and strength
                we = self.get_edited_latent(w, d, [strength])[0]
                direction = we - w
                direction = direction.mean(0, keepdim=True)  # Shape for fusion and losses

                # First forward pass: Decoding w and we
                x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, return_latents=False, return_7=True, feats=None)
                xe, _  = self.net.decoder([we], input_is_latent=True, randomize_noise=False, return_latents=False, return_7=True, feats=None)

                # Post-processing: Clamping and pooling
                xe.clamp_(-1, 1)
                x.clamp_(-1, 1)
                x_input = self.face_pool(x)
                xe_input = self.face_pool(xe)

            # Second forward pass: Encoding x and decoding with adjusted latent code
            w_hat, F_hat = self.net.encoder(x_input)
            w_hat += self.net.latent_avg[None, ...]
            F_fu = self.net.fuser(F_hat, direction)
            y_hat, _ = self.net.decoder([w_hat + direction], input_is_latent=True, randomize_noise=True, return_latents=False, return_7=True, feats=F_fu)
            y_hat.clamp_(-1, 1)
            y_hat_output = self.face_pool(y_hat)

            # Loss calculation
            self.optimizer.zero_grad()

            # Identity loss (reconstruction of edited image)
            loss_edit, loss_dict, id_logs = self.calc_loss(xe_input, y_hat_output, task_type='editability')
            l2_multiscale = multiscale_l2_loss(y_hat, xe) * self.opts.l2_lambda_edit
            loss_edit += l2_multiscale

            # Latent code loss to ensure the edits are preserved and consistent
            we_hat, _ = self.net.encoder(y_hat_output)

            # Editability (direction consistency) loss: Ensures that applying the same direction to w produces consistent results
            direction_loss = torch.mean(torch.abs((we_hat - w_hat) - direction)) * self.opts.direction_lambda

            # Total variation loss to encourage smoothness in the output
            tv_loss = total_variation_loss(F_fu) * 4

            if self.global_step % self.opts.Inversion_every == 0:

                # Cycle consistency loss: Reconstruct original image x from the latent w_hat
                F_x = self.net.fuser(F_hat, direction * 0.0)  # Without direction for inversion
                x_hat, _ = self.net.decoder([w_hat], input_is_latent=True, randomize_noise=True, return_latents=False, return_7=True, feats=F_x)
                x_hat.clamp_(-1, 1)
                x_hat_output =  self.face_pool(x_hat).clamp(-1, 1)

                loss_inv, dict_Inv, _ = self.calc_loss(x_input, x_hat_output, task_type='inversion')
                dict_Inv = {key + '_Inv': value for key, value in dict_Inv.items()}
                l2_multiscale_inv = multiscale_l2_loss(x_hat, x) * self.opts.l2_lambda_inv 
                loss_inv += l2_multiscale_inv

                tv_loss_inv = total_variation_loss(F_x) * 4
                loss_inv += tv_loss_inv

                # Combine losses for backward pass
                total_loss = loss_edit + tv_loss + direction_loss + loss_inv

                # Backward pass and optimization
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                dict_Inv['tv_loss_Inv'] = tv_loss_inv.item()
                loss_dict.update(dict_Inv)

                # Update loss dictionary for logging
                loss_dict['total_loss'] = total_loss.item()
                loss_dict['direction_loss'] = direction_loss.item()
                loss_dict['tv_loss'] = tv_loss.item()
                loss_dict['inv_loss'] = loss_inv.item()
                loss_dict['l2_ms'] = l2_multiscale.item()
                loss_dict['l2_ms_inv'] = l2_multiscale_inv.item()

            else:
                # Combine losses for backward pass
                total_loss = loss_edit + tv_loss + direction_loss  

                # Backward pass and optimization
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Update loss dictionary for logging
                loss_dict['total_loss'] = total_loss.item()
                loss_dict['direction_loss'] = direction_loss.item()
                loss_dict['tv_loss'] = tv_loss.item()
                loss_dict['l2_ms'] = l2_multiscale.item()

            # Logging related
            if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
                if self.wandb_logger:
                    self.parse_and_log_images(id_logs, x_input, xe_input, y_hat_output, title='images/train/faces')
            if self.global_step % self.opts.board_interval == 0:
                self.print_metrics(loss_dict, prefix='train')
                if self.wandb_logger:
                    self.log_metrics(loss_dict, prefix='train')

            # Validation related
            val_loss_dict = None
            if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                val_loss_dict = self.validate()
                if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                    self.best_val_loss = val_loss_dict['loss']
                    self.checkpoint_me(val_loss_dict, is_best=True)

            if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                if val_loss_dict is not None:
                    self.checkpoint_me(val_loss_dict, is_best=False)
                else:
                    self.checkpoint_me(loss_dict, is_best=False)

            if self.global_step == self.opts.max_steps:
                print('OMG, finished training!')
                break

            self.global_step += 1


    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            x, y = batch

            with torch.no_grad():

                x, y = x.to(self.device).float(), y.to(self.device).float()

                w, F0 = self.net.encoder(x)
                w += self.net.latent_avg[None, ...]
                d, strength = "fs_glasses", 18
                we = self.get_edited_latent(w, d, [strength])[0]
                direction = we - w
                direction = direction[0, None, ...]  # Shape for fusion and losses

                F_fu = self.net.fuser(F0, direction)
                y_hat, _ = self.net.decoder([w + direction], input_is_latent=True, randomize_noise=True, return_latents=False, return_7=True, feats=F_fu)
                y_hat_1 = self.face_pool(y_hat)

                F_fu = self.net.fuser(F0, direction*0)
                y_hat, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=True, return_latents=False, return_7=True, feats=F_fu)
                y_hat = self.face_pool(y_hat)
                loss, cur_loss_dict, id_logs = self.calc_loss(y, y_hat, task_type='test')
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            if batch_idx <= 1:
                self.parse_and_log_images(id_logs, x, y_hat, y_hat_1, title='images/test/faces', subscript='{:04d}'.format(batch_idx))

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:

                self.net.train()
                if not self.opts.train_decoder:
                    self.net.decoder.eval()
                else:
                    self.net.decoder.train()
                
                return None

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        if not self.opts.train_decoder:
            self.net.decoder.eval()
        else:
            self.net.decoder.train()

        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'latest_model.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        
        if hasattr(self.net, 'fuser'):
            print('\nAdding Fuser parameters to the training phase..........\n')
            params = [{'params': self.net.encoder.parameters(), 'lr': self.opts.lr_encoder}, {'params': self.net.fuser.parameters()}]
        else:
            params = [{'params': self.net.encoder.parameters()}]

        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        else:
            for param in self.net.decoder.parameters():
                param.requires_grad = False

        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.AdamW(params, lr=self.opts.lr_fuser)
        else:
            optimizer = Ranger(params, lr=self.opts.lr_fuser)

        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset_celeba = ImagesDataset(source_root=dataset_args['train_source_root'],
                                             target_root=dataset_args['train_target_root'],
                                             source_transform=transforms_dict['transform_source'],
                                             target_transform=transforms_dict['transform_gt_train'],
                                             opts=self.opts)
        test_dataset_celeba = ImagesDataset(source_root=dataset_args['test_source_root'],
                                            target_root=dataset_args['test_target_root'],
                                            source_transform=transforms_dict['transform_source'],
                                            target_transform=transforms_dict['transform_test'],
                                            opts=self.opts, test=True)
        train_dataset = train_dataset_celeba
        test_dataset = test_dataset_celeba
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset


    def calc_loss(self, y, y_hat, task_type='inversion'):

        # Use parsed weights based on the task type (inversion or editability)
        if task_type == 'inversion':
            lpips_lambda = self.opts.lpips_lambda_inv
            id_lambda = self.opts.id_lambda_inv
            parse_lambda = self.opts.parse_lambda_inv
            l2_lambda = self.opts.l2_lambda_inv
        elif task_type == 'editability':
            lpips_lambda = self.opts.lpips_lambda_edit
            id_lambda = self.opts.id_lambda_edit
            parse_lambda = self.opts.parse_lambda_edit
            l2_lambda = self.opts.l2_lambda_edit
        elif task_type == 'test':
            lpips_lambda = 2
            id_lambda = 2
            parse_lambda = 2
            l2_lambda = 2
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Initialize loss and loss dictionary
        loss_dict = {}
        loss = 0.0
        id_logs = None

        # Identity loss
        if id_lambda > 0:
            loss_id = self.id_loss(y_hat, y) * id_lambda
            loss_dict['loss_id'] = float(loss_id)
            loss += loss_id

        # Face Parsing loss
        if parse_lambda > 0:
            loss_parse = self.parse_loss(y_hat, y) * parse_lambda
            loss_dict['loss_parse'] = float(loss_parse)
            loss += loss_parse

        # MSE (L2) loss
        if l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y) * l2_lambda
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2

        # Perceptual loss (LPIPS)
        if lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y) * lpips_lambda
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips

        # Total loss
        loss_dict['loss'] = float(loss)

        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):

        wandb.log({prefix + '/' + key: value for key, value in metrics_dict.items()}, step=self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)

        if subscript is not None:
            name_1 = 'Imgs/' + name.split('/')[1] + '_' + subscript[-1]
            wandb.log({name_1: plt}, step=self.global_step)
        else:
            name_1 = 'Imgs/' + name.split('/')[1]
            wandb.log({name_1: plt}, step=self.global_step)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg
        return save_dict
    
    def get_edited_latent(self, original_latent, editing_name, editing_degrees, original_image=None):
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
        return edited_latents

def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = \
        torch.autograd.grad(outputs=d_out.sum(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True,
                            allow_unused=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

def total_variation_loss(img):
    # Calculate differences between neighboring pixels
    diff_x = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_y = img[:, :, :, 1:] - img[:, :, :, :-1]

    # Sum of squared differences
    loss = torch.mean(torch.abs(diff_x)) + torch.mean(torch.abs(diff_y))

    return loss


def multiscale_l2_loss(estimated, ground_truth):
    """
    Computes the multiscale L2 loss between estimated and ground truth images at
    resolutions of 256x256, 512x512, and 1024x1024.
    
    Args:
        estimated (torch.Tensor): The estimated output image, assumed to be of shape (B, C, H, W) with 1024x1024 resolution.
        ground_truth (torch.Tensor): The ground truth image, assumed to be of shape (B, C, H, W) with 1024x1024 resolution.

    Returns:
        torch.Tensor: The computed multiscale L2 loss.
    """
    # Ensure both images are at same resolution
    assert estimated.shape == ground_truth.shape, "Both images should be of same shape"
    
    # Initialize total loss
    total_loss = 0.0

    # Define scales and corresponding resolutions
    scales = [512, 768, 1024]
    
    for scale in scales:
        # Downsample images to the current scale
        estimated_resized = F.interpolate(estimated, size=(scale, scale), mode='bilinear', align_corners=False)
        ground_truth_resized = F.interpolate(ground_truth, size=(scale, scale), mode='bilinear', align_corners=False)
        
        # Compute L2 loss for the current scale
        scale_loss = F.mse_loss(estimated_resized, ground_truth_resized)
        
        # Add to total loss
        total_loss += scale_loss

    return total_loss