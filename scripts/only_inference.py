import os
from argparse import Namespace
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
import json
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets_1.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.inv_mamba import Mamba_Inv
import pdb

torch.cuda.set_device(0)
device = torch.device("cuda:0")

"""FACE_DIRECTIONS = {
    "age": np.arange(5, 7.1, 0.5)}"""

"""FACE_DIRECTIONS = {
    "age": [-7, -5, -3, 3, 5, 7, 10],
    "fs_makeup": [5, 8, 10, 12],
    "afro": [0.1, 0.15, 0.18, 0.2],
    "angry": [0.125, 0.15, 0.2],
    "purple_hair": [0.1, 0.15, 0.2],
    "fs_glasses": [8, 10, 12, 14, 16, 18, 20, 22],
    "face_roundness": [-13, -7, 7, 13],
    "rotation": [-7.0, -5.0, -3.0, 3.0, 5.0, 7.0],
    "bobcut": [0.12, 0.16, 0.18, 0.2, 0.23],
    "bowlcut": [0.11, 0.15, 0.18, 0.2, 0.22],
    "mohawk": [0.11, 0.15, 0.18, 0.2, 0.23],
    "fs_smiling": [-6, -4,-2, 2, 4, 6, 8, 10],
}"""

FACE_DIRECTIONS = {
    "Inversion": [0],
    "age": np.arange(-7, 12.1, 1),
    "fs_makeup": np.arange(5, 11, 2),
    "afro": np.arange(0.1, 0.21, 0.2),
    "angry": np.arange(0.125, 0.21, 0.2),
    "purple_hair": np.arange(0.1, 0.31, 0.05),
    "fs_glasses": np.arange(8, 23, 2),
    "face_roundness": np.arange(-13, 13, 4),
    "rotation": np.arange(-7.0, 8.0, 2.0),
    "bobcut": np.arange(0.12, 0.24, 0.2),
    "bowlcut": np.arange(0.11, 0.23, 0.2),
    "mohawk": np.arange(0.11, 0.24, 0.2),
    "fs_smiling": np.arange(-6, 11, 2),
}

CAR_DIRECTIONS = {
    # "Inversion": [0],
    # "color": [-22, -18, -15, -12, 12, 15, 18, 22],
    # "grass": [-40, -35, -30, -25, -20, 20, 25, 30, 35, 40],
    "pose_1": [-0.1, -0.3, -0.6, 0.1, 0.3, 0.6],
    "pose_2": [-3, -1, 1, 3],
    "cube": [-13, -10, -5,  5, 10, 13]
}

def inverse_normalize_batch(batch_tensor, mean=None, std=None):
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]
    for t in batch_tensor:
        for channel, m, s in zip(t, mean, std):
            channel.mul_(s).add_(m)
    return batch_tensor


def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image


def save_image_as_svg(image, save_path):
    # Convert image to matplotlib format and save as SVG
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')  # Turn off axis
    plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0)
    plt.close()


def run():
    test_opts = TestOptions().parse()

    # Load options from checkpoint
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    try:
        opts = ckpt['opts']
    except:
        with open('./opts.json', 'r') as file:
            opts = json.load(file)

    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False

    opts = Namespace(**opts)
    opts.device= device
    opts.start_from_latent_avg = True

    net = Mamba_Inv(opts)
    net.eval()
    net.cuda()

    print(f'Loading dataset for {opts.dataset_type}')
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path, transform=transforms_dict['transform_inference'], opts=opts)
    dataloader = DataLoader(dataset, batch_size=opts.test_batch_size, shuffle=False, num_workers=int(opts.test_workers), drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

        # Process each direction
        for direction_name, strengths in FACE_DIRECTIONS.items():
            print(f"\nWorking in Direction: {direction_name}")
            out_path_results = os.path.join(test_opts.exp_dir, f"inference_{direction_name}")
            os.makedirs(out_path_results, exist_ok=True)

            global_i = 0
            for input_batch in tqdm(dataloader):
                with torch.no_grad():
                    input_cuda = input_batch.cuda().float()

                    for strength in strengths:
                        strength_path = os.path.join(out_path_results, f"strength_{strength}")
                        os.makedirs(strength_path, exist_ok=True)

                        result_batch, _ = net(input_cuda, randomize_noise=False, resize=opts.resize_outputs, return_latents=True, strength=strength, direction_name=direction_name)

                        for i in range(input_batch.shape[0]):
                            result = tensor2im(result_batch[i], is_train=True)
                            im_path = dataset.paths[global_i + i]
                            result_save_path = os.path.join(strength_path, os.path.basename(im_path))
                            result.save(result_save_path)

                global_i += input_batch.shape[0]


if __name__ == '__main__':
    run()
