from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, default=None, help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str, help='Type of dataset/experiment to run')
        self.parser.add_argument('--training_stage', default=1, type=int, help='Training the E2Style encoder for stage i')
        self.parser.add_argument('--is_training', default=True, type=bool, help='Training or testing')
        self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the E2Style encoder')
        self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the E2Style encoder')

        self.parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=4, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=4, type=int, help='Number of test/inference dataloader workers')

        self.parser.add_argument('--lr_encoder', default=5e-4, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--lr_fuser', default=5e-4, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--gamma_scheduler', default=0.1, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='adam', type=str, help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', default=True, action='store_true', help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space insteaf of w+')

        self.parser.add_argument('--lpips_lambda_inv', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda_inv', default=0.1, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--parse_lambda_inv', default=1.0, type=float, help='Mulit-Parse loss multiplier factor')
        self.parser.add_argument('--l2_lambda_inv', default=5.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--w_norm_lambda', default=0.005, type=float, help='W-norm loss multiplier factor')

        self.parser.add_argument('--lpips_lambda_edit', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda_edit', default=0.1, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--parse_lambda_edit', default=1.0, type=float, help='Mulit-Parse loss multiplier factor')
        self.parser.add_argument('--l2_lambda_edit', default=5.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--direction_lambda', default=2.0, type=float, help='direction loss multiplier factor')

        self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str, help='Path to StyleGAN model weights')
        self.parser.add_argument('--checkpoint_path', default=None, # "./Results/Model_1/checkpoints/best_model.pt",
                                 type=str, help='Path to model checkpoint')

        self.parser.add_argument('--max_steps', default=510005, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=500, type=int, help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=5000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=1000, type=int, help='Model checkpoint interval')
        self.parser.add_argument('--Inversion_every', default=25, type=int, help='Make inversion procces every K steps')

        # arguments for super-resolution
        self.parser.add_argument('--resize_factors', type=str, default=None,
                                 help='For super-res, comma-separated resize factors to use for inference.')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
