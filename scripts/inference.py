import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets_1.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.inv_mamba import Mamba_Inv
from criteria.Quality import ssim, ms_ssim, psnr
import torch.nn.functional as F
from criteria.lpips.lpips import LPIPS
from criteria.FID import calculate_fid_given_paths
import pdb
from calflops import calculate_flops
import json


def inverse_normalize_batch(batch_tensor, mean=None, std=None):
    # Assuming batch_tensor is of shape (batch_size, channels, height, width)
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]
    for t in batch_tensor:
        for channel, m, s in zip(t, mean, std):
            channel.mul_(s).add_(m)
    return batch_tensor


def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    try:
        opts = ckpt['opts']
    except:
        with open('./opts.json', 'r')  as file:
            opts = json.load(file)

    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    opts = Namespace(**opts)

    net = Mamba_Inv(opts)
    # net = E2Style(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    total_params = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad and not 'decoder' in n)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []
    latent_list = []
    psnr_list = []
    ssim_list = []
    ms_ssim_list = []
    mse_list = []
    lpips_list = []
    lpips_metric = LPIPS(net_type='alex').to('cuda').eval()
    image_list_path = os.path.join(opts.exp_dir, 'image_list.txt')

    _, macs, _ = calculate_flops(model=net, input_shape=(1, 3, 256, 256), output_as_string=True, output_precision=4, print_results=False)

    with open(image_list_path, 'w') as f:
        for input_batch in tqdm(dataloader):
            with torch.no_grad():
                input_cuda = input_batch.cuda().float()
                tic = time.time()
                x = F.interpolate(input_cuda, size=(256, 256), mode="bilinear", align_corners=False)
                result_batch, latent_batch = net(x, randomize_noise=True, resize=opts.resize_outputs, return_latents=True)
                x_hat = F.interpolate(result_batch, size=(256, 256), mode="bilinear", align_corners=False)
                latent_list.append(latent_batch)
                toc = time.time()
                global_time.append(toc - tic)
                lpips_list.append(lpips_metric(x_hat, x).item())
                inverse_normalize_batch(result_batch).clamp_(0, 1)
                inverse_normalize_batch(input_cuda).clamp_(0, 1)
                ssim_ = ssim(result_batch, input_cuda, size_average=False, data_range=1.0)
                psnr_ = psnr(result_batch, input_cuda, max_val=1)
                ms_ssim_ = ms_ssim(result_batch, input_cuda, size_average=False, data_range=1.0)
                mse_list.append(F.mse_loss(result_batch, input_cuda).item())

            if True:
                for i in range(input_batch.shape[0]):
                    result = tensor2im(result_batch[i], is_train=False)
                    im_path = dataset.paths[global_i]
                    f.write(im_path + '\t PSNR: {:.4f} \t SSIM: {:.4f} \t MS-SSIM: {:.4f}'.format(psnr_[i].item(), ssim_[i].item(), ms_ssim_[i].item()) + '\r\n')
                    ssim_list.append(ssim_[i].item())
                    psnr_list.append(psnr_[i].item())
                    ms_ssim_list.append(ms_ssim_[i].item())

                    if opts.couple_outputs or global_i % 100 == 0:
                        input_im = log_input_image(input_batch[i], opts)
                        resize_amount = (256, 256) if opts.resize_outputs else (1024, 1024)
                        if opts.resize_factors is not None:
                            # for super resolution, save the original, down-sampled, and output
                            source = Image.open(im_path)
                            res = np.concatenate([np.array(source.resize(resize_amount)),
                                                  np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
                                                  np.array(result.resize(resize_amount))], axis=1)
                        else:
                            # otherwise, save the original and output
                            res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                                  np.array(result.resize(resize_amount))], axis=1)
                        Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

                    im_save_path = os.path.join(out_path_results, os.path.basename(im_path))# .replace(".jpg", ".png")
                    Image.fromarray(np.array(result)).save(im_save_path)

                    global_i += 1

    f.close()

    fid_value = calculate_fid_given_paths(paths=[opts.exp_dir + '/inference_results/',
                                                 opts.data_path], img_size=256, batch_size=16)
    stats_path = os.path.join(opts.exp_dir, 'stats_inversion.txt')
    if opts.save_inverted_codes:
        np.save(os.path.join(opts.exp_dir, f"latent_code.npy"), torch.cat(latent_list, 0).detach().cpu().numpy())
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str + '\r\n')
        f.write('PSNR: {:.4f}'.format(sum(psnr_list) / psnr_list.__len__()) + '\r\n')
        f.write('SSIM: {:.4f}'.format(sum(ssim_list) / ssim_list.__len__()) + '\r\n')
        f.write('MS-SSIM: {:.4f}'.format(sum(ms_ssim_list) / ms_ssim_list.__len__()) + '\r\n')
        f.write('MSE: {:.4f}'.format(sum(mse_list) / mse_list.__len__()) + '\r\n')
        f.write('LPIPS: {:.4f}'.format(sum(lpips_list) / lpips_list.__len__()) + '\r\n')
        f.write('FID: {:.4f}'.format(fid_value) + '\r\n')
        f.write('Total Parameters: {:.1f}'.format(total_params) + '\r\n')
        f.write('{:<8}  {:<8}'.format('Computational complexity:', macs) + '\r\n')
        if 'global_step' in ckpt:
            f.write('Total Iters: {:.1f}'.format(ckpt['global_step']) + '\r\n')

    f.close()


if __name__ == '__main__':
    run()
