from argparse import ArgumentParser


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # arguments for inference script
        self.parser.add_argument('--exp_dir', type=str, default='./Final_Results/Model_3/test/', help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path', default='./Final_Results/Model_3/checkpoints/latest_model.pt', type=str, help='Path to E2Style model checkpoint')
        self.parser.add_argument('--stage', default=1, type=int, help='Results of stage i')
        self.parser.add_argument('--is_training', default=False, type=bool, help='Training or testing')
        self.parser.add_argument('--data_path', type=str, default= './data/celeba_hq/test/', # './data/celeba_hq/test/', # './data/Stanford_Cars/cars_test/'
                                 help='Path to directory of images to evaluate')
        self.parser.add_argument('--couple_outputs', action='store_true', default=False, help='Whether to also save inputs + outputs side-by-side')
        self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs to 256x256 or keep at 1024x1024')
        self.parser.add_argument('--save_inverted_codes', action='store_true', default=True, help='Whether to save the inverted latent codes')

        self.parser.add_argument('--test_batch_size', default=32, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=8, type=int, help='Number of test/inference dataloader workers')

        self.parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')


    def parse(self):
        opts = self.parser.parse_args()
        return opts
