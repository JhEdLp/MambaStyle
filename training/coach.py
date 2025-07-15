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
    "age":np.arange(-7, 12.1, 1),
    "fs_glasses": np.arange(9, 23, 2),
    "rotation": np.arange(-7.0, 9, 1),
    "purple_hair": np.arange(0.09, 0.25, 0.02),
    "fs_smiling": np.arange(-6, 9, 2),
    "afro":  np.arange(0.09, 0.31, 0.03),
    "fs_makeup": [5, 8, 10, 12],
    "angry": np.arange(0.09, 0.27, 0.25),
    "face_roundness": [-13, -7, 7, 13],
    "bobcut": np.arange(0.1, 0.21, 0.02),
    "bowlcut": np.arange(0.1, 0.21, 0.02),
    "mohawk": np.arange(0.1, 0.21, 0.02),
}

CAR_DIRECTIONS = {
    "pose_1": [-3, -1, 1, 3],
    "pose_2": [-3, -1, 1, 3],
    "cube": [-20, -15, 10, 10, 15, 20],
    "color": [-22, -18, -15, -12, 12, 15, 18, 22],
    "grass": [-40, -33, -27, -20],
}

# Define the order in which attributes should be unlocked
UNLOCK_ORDER = list(FACE_DIRECTIONS.keys())  # Maintain a fixed order

def get_random_edit(iteration=1e3):
    """Returns a random direction and strength, unlocking more attributes every 100 iterations."""
    
    unlock_index = min(len(UNLOCK_ORDER), (iteration // 1000) + 1)  # Increase visible keys every 100 iterations
    visible_keys = UNLOCK_ORDER[:unlock_index]  # Get only the unlocked attributes

    direction = np.random.choice(visible_keys)  # Pick a direction from unlocked keys
    strength = np.random.choice(FACE_DIRECTIONS[direction])  # Pick a strength from the chosen direction
    
    return direction, strength

'''def get_random_edit():
    direction = np.random.choice(list(FACE_DIRECTIONS.keys()))
    strenght = np.random.choice(FACE_DIRECTIONS[direction])
    return direction, strenght'''


class Coach:
    def __init__(self, opts):

        try:
            import wandb
            args_wandb = vars(opts)
            wandb.init(name='Model_1', dir='../', project='Mamba2_Gan_Inv', config=args_wandb, id='93q0fgjg', resume="must",
                       notes='Using Direction Loss and increasing the number of directions along iterations')
            self.wandb_logger = True

            # Create an artifact
            artifact = wandb.Artifact("Coach", type="code")
            # Add the Model file to the artifact
            artifact.add_file('./training/coach.py')
            # Log the artifact
            wandb.log_artifact(artifact)
            
        except ImportError:
            print("WandB is not installed. Please install it using 'pip install wandb'.")
            # raise SystemExit(1)
            self.wandb_logger = False

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
        if self.opts.w_norm_lambda > 0:
            self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)

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
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=40000, gamma=self.opts.gamma_scheduler)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,     T_0=15000,     T_mult=2,     eta_min=1e-6 ) 

        if self.net.global_step is not None:
            self.global_step = self.net.global_step
            # pass

        if self.net.optimizer_ckpt is not None:
            self.optimizer.load_state_dict(self.net.optimizer_ckpt)
            # pass

        self.resizer = train_utils.ImageResizer(size=(256, 256), mode='bilinear', antialias=True) # torch.nn.AdaptiveAvgPool2d((256, 256))

        self.latent_editor = LatentEditor("car" if "car" in opts.dataset_type else "human_faces")

    def Inversion_step(self, direction, x=None):

        if x is not None:
            batch = next(iter(self.train_dataloader))
            x, _ = batch
            x = x.to(self.device).float()

        x_input = self.resizer.interpolate(x)

        w_hat, F_hat = self.net.encoder(x_input)
        w_hat = w_hat + self.net.latent_avg[None, ...]
        F_fu = self.net.fuser(F_hat, direction)
        x_hat, _ = self.net.decoder([w_hat], input_is_latent=True, randomize_noise=True, return_latents=False, return_7=True, feats=F_fu)
        x_hat.clamp_(-1, 1)
        x_hat_output = self.resizer.interpolate(x_hat)

        loss_inv, dict_Inv, _ = self.calc_loss(x_input, x_hat_output, task_type='inversion', latent=w_hat)
        dict_Inv = {key + '_Inv': value for key, value in dict_Inv.items()}

        l2_multiscale_inv = multiscale_l2_loss(x_hat, x) * self.opts.l2_lambda_inv
        f_norm =  F.smooth_l1_loss(F_fu, F_hat)# torch.norm(F_fu, p='fro').detach() * 5e-4

        tv_loss_inv = total_variation_loss(F_fu).detach() * 4

        total_loss = loss_inv + l2_multiscale_inv + f_norm   

        # Update loss dictionary for logging
        dict_Inv['loss_Inv'] = total_loss.item()
        dict_Inv['tv_loss_Inv'] = tv_loss_inv.item()
        dict_Inv['l2_ms_inv'] = l2_multiscale_inv.item()

        return total_loss, dict_Inv,  [x, x_input, x_hat, x_hat_output]
    
    def Edition_step(self):
        """
        Performs an edition step in the latent space, applies a random direction edit, 
        and computes various losses to ensure editability and reconstruction consistency.
        """

        # === Step 1: Generate and Edit Latent Codes ===
        with torch.no_grad():
            # Sample latent codes from Gaussian noise
            latent_noise = torch.randn([self.opts.batch_size, 18, 512], device=self.device)
            w = self.net.decoder.style(latent_noise)  # Map z to W space
            w = w + self.net.latent_avg[None, ...]    # Centering around latent mean

            # Generate random edit direction
            d, strength = get_random_edit(self.global_step)  
            we = self.get_edited_latent(w, d, [strength])[0]
            direction = (we - w).detach().mean(0, keepdim=True)  # Ensure direction is detached

            # Decode both original and edited latents
            x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, return_7=True)
            xe, F9_ed = self.net.decoder([we], input_is_latent=True, randomize_noise=False, return_7=True)

            # Apply clamping to keep pixel values in a valid range
            x.clamp_(-1, 1)
            xe.clamp_(-1, 1)

            # Resize for encoder input
            x_input = self.resizer.interpolate(x)
            xe_input = self.resizer.interpolate(xe)

        # === Step 2: Encode and Apply Edit Direction ===
        w_hat, F_hat = self.net.encoder(x_input)  # Encode original image
        w_hat = w_hat + self.net.latent_avg[None, ...]  # Center around latent mean

        # Fuse features using edit direction
        F_fu = self.net.fuser(F_hat, direction)

        # Decode edited latent + features
        y_hat, _ = self.net.decoder([w_hat + direction], input_is_latent=True, randomize_noise=False, return_7=True, feats=F_fu)
        y_hat_output = self.resizer.interpolate(y_hat)

        # === Step 3: Compute Editability Losses ===
        loss_edit, loss_dict, id_logs = self.calc_loss(xe_input, y_hat_output, task_type='editability', latent=w_hat)

        # Feature space regularization
        f_norm = F.smooth_l1_loss(F_fu, F9_ed)  # Ensuring features remain aligned
        l2_multiscale = multiscale_l2_loss(y_hat, xe) * self.opts.l2_lambda_edit

        # === Step 4: Cycle Consistency Check ===
        we_hat, F_aux = self.net.encoder(y_hat_output)  
        we_hat = we_hat + self.net.latent_avg[None, ...]  

        # Reverse edit to reconstruct original features
        F_x = self.net.fuser(F_aux, direction * 0.0)  
        x_hat, _ = self.net.decoder([we_hat - direction], input_is_latent=True, randomize_noise=False, return_7=True, feats=F_x)
        x_hat_output = self.resizer.interpolate(x_hat)
        loss_inv, dict_Inv, _ = self.calc_loss(x_input, x_hat_output, task_type='inversion', latent=None)
        dict_Inv = {key + '_Inv': value for key, value in dict_Inv.items()}

        # Compute cycle consistency loss (ensures reversibility of edits)
        cycle_loss = multiscale_l2_loss(x_hat, x) * self.opts.l2_lambda_inv

        # === Step 5: Direction Consistency Loss ===
        w_hat_norm = (w_hat - self.net.latent_avg) / (w_hat.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        we_hat_norm = (we_hat - self.net.latent_avg) / (we_hat.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        direction_norm = direction / (direction.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        direction_loss = torch.mean(torch.abs((we_hat_norm - w_hat_norm) - direction_norm)) * self.opts.direction_lambda

        # === Step 6: Regularization Losses ===
        tv_loss = total_variation_loss(x_hat) * 4 + total_variation_loss(y_hat) * 4  # Encourages smoothness

        # Compute total loss
        total_loss = loss_edit + l2_multiscale + f_norm + tv_loss + cycle_loss

        # Update loss dictionary for logging
        loss_dict.update({
            'direction_loss': direction_loss.item(),
            'tv_loss': tv_loss.item(),
            'l2_ms': l2_multiscale.item(),
            'Feats_norm': f_norm.item(),
            'Cycle_Loss': cycle_loss.item(),
        })
        loss_dict.update(dict_Inv)
        # pdb.set_trace()

        return total_loss, loss_dict, [x_input, xe_input, y_hat_output, y_hat, x]

    
    def train_jhon(self):
        
        self.net.eval()
        self.net.encoder.train()
        self.net.fuser.train()

        if not self.opts.train_decoder:
            self.net.decoder.eval()
        else:
            self.net.decoder.train()

        inv_dir = torch.zeros((1, 18, 512), device=self.device)
        
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.real_labels = torch.ones(self.opts.batch_size, 1).to(self.device)
        self.fake_labels = torch.zeros(self.opts.batch_size, 1).to(self.device)

        while self.global_step < self.opts.max_steps:

            self.optimizer.zero_grad()

            # Edition Step
            total_loss, loss_dict, attr = self.Edition_step()
            # D_edit = self.net.discriminator(attr[-1]) # Edited images
            # loss_adv_edit = self.adv_loss(D_edit, self.real_labels)  # Fool discriminator for edited images
            # total_loss += loss_adv_edit * 0.5

            # Backward pass
            # total_loss.backward()

            # loss_dict["Adv_Edit"] = loss_adv_edit.item()
            # loss_dict["Adv_Inv"] = loss_adv_inv.item()


            if self.global_step % self.opts.Inversion_every == 0:

                # Inversion Step
                loss_inv, inv_dict, attr_1 = self.Inversion_step(direction=inv_dir, x=attr[-1])
                # D_inv = self.net.discriminator(attr_1[2])

                # Adversarial loss for inverted images
                # loss_adv_inv = self.adv_loss(D_inv, self.real_labels)  # Fool discriminator for inverted images

                # Combine inversion loss and adversarial loss
                # loss_inv += loss_adv_inv * 0.5

                # Backward pass and optimization
                # loss_inv.backward()
                loss_dict.update(inv_dict)
                total_loss += loss_inv

            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Logging related
            if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
                if self.wandb_logger:
                    self.parse_and_log_images(None, attr[0], attr[1], attr[2], title='images/train/faces')
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
                w = w + self.net.latent_avg[None, ...]
                d, strength = "fs_glasses", 18
                we = self.get_edited_latent(w, d, [strength])[0]
                direction = we - w
                direction = direction[0, None, ...]  # Shape for fusion and losses

                F_fu = self.net.fuser(F0, direction)
                y_hat, _ = self.net.decoder([we], input_is_latent=True, randomize_noise=True, return_latents=False, return_7=True, feats=F_fu)
                y_hat_1 = self.resizer.interpolate(y_hat)

                F_fu = self.net.fuser(F0, direction*0)
                y_hat, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=True, return_latents=False, return_7=True, feats=F_fu)
                y_hat = self.resizer.interpolate(y_hat)
                loss, cur_loss_dict, id_logs = self.calc_loss(y, y_hat, task_type='inversion')
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            if batch_idx <= 1:
                self.parse_and_log_images(id_logs, x, y_hat, y_hat_1, title='images/test/faces', subscript='{:04d}'.format(batch_idx))

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        # self.net.discriminator.eval()
        if not self.opts.train_decoder:
            self.net.decoder.eval()
        else:
            self.net.decoder.train()

        if self.global_step == 0:
            return None
        
        else: 
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
            params = [{'params': self.net.fuser.parameters(), 'lr': self.opts.lr_fuser}, {'params': self.net.encoder.parameters()},]
                      # {'params': self.net.discriminator.parameters(), 'lr': 3e-5}]
        else:
            params = [{'params': self.net.encoder.parameters()}]

        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        else:
            for param in self.net.decoder.parameters():
                param.requires_grad = False
            
            # for param in self.net.discriminator.parameters():
            #     param.requires_grad = False

        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.AdamW(params, lr=self.opts.lr_encoder)
        else:
            optimizer = Ranger(params, lr=self.opts.lr_encoder)

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


    def calc_loss(self, y, y_hat, task_type='inversion', latent=None):

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
		
        if self.opts.w_norm_lambda > 0 and latent is not None:
            loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg) * self.opts.w_norm_lambda
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm 

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
            'opts': vars(self.opts),
            'global_step': self.global_step
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg
        return save_dict
    
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
        scale_loss = F.smooth_l1_loss(estimated_resized, ground_truth_resized)
        
        # Add to total loss
        total_loss += scale_loss

    return total_loss


def compute_gram_matrix(features):
    """
    Compute the Gram matrix (F F^T) for feature alignment.
    Args:
        features (torch.Tensor): Feature tensor of shape (B, C, H, W)
    Returns:
        torch.Tensor: Gram matrix of shape (B, HW, HW)
    """
    B, C, H, W = features.shape
    features = features.view(B, C, H * W)  # Reshape to (B, C, HW)
    gram_matrix = torch.bmm(features, features.transpose(1, 2))  # Compute F F^T
    return gram_matrix

def cka_loss(student_features, teacher_features):
    """
    Compute the Centered Kernel Alignment (CKA) loss.
    Args:
        student_features (torch.Tensor): Student model feature maps (B, C, H, W)
        teacher_features (torch.Tensor): Teacher model feature maps (B, C, H, W)
    Returns:
        torch.Tensor: CKA loss value
    """
    K_student = compute_gram_matrix(student_features)
    K_teacher = compute_gram_matrix(teacher_features)
    
    # Frobenius norm
    norm_student = torch.norm(K_student, dim=(1, 2))
    norm_teacher = torch.norm(K_teacher, dim=(1, 2))
    
    # Compute CKA loss
    cka_similarity = torch.sum(K_student * K_teacher, dim=(1, 2)) / (norm_student * norm_teacher + 1e-8)
    loss = 1 - cka_similarity.mean()
    
    return loss