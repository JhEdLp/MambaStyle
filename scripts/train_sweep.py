"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach_sweep import Coach
import wandb
import random

# List of 50 adjectives
adjectives = [
    'quick', 'silent', 'flying', 'brave', 'bright', 'clever', 'lucky', 'fierce', 'mighty', 'swift',
    'fearless', 'wild', 'calm', 'bold', 'daring', 'happy', 'strong', 'gentle', 'sharp', 'smart',
    'curious', 'eager', 'proud', 'loyal', 'honest', 'wise', 'mysterious', 'playful', 'graceful', 'sneaky',
    'vigilant', 'trusty', 'radiant', 'shiny', 'noble', 'diligent', 'courageous', 'fearsome', 'victorious', 'peaceful',
    'alert', 'silent', 'glowing', 'serene', 'thunderous', 'stormy', 'boundless', 'endless', 'fiery', 'blazing'
]

# List of 50 nouns
nouns = [
    'panther', 'tiger', 'falcon', 'eagle', 'lion', 'shark', 'dragon', 'wolf', 'fox', 'hawk',
    'bear', 'cobra', 'cheetah', 'whale', 'jaguar', 'raven', 'rhino', 'serpent', 'stallion', 'cougar',
    'bison', 'buffalo', 'antelope', 'viper', 'lynx', 'sparrow', 'orca', 'python', 'coyote', 'leopard',
    'gazelle', 'hyena', 'pelican', 'scorpion', 'otter', 'octopus', 'dolphin', 'walrus', 'mongoose', 'falcon',
    'hedgehog', 'eagle', 'owl', 'falcon', 'rabbit', 'badger', 'fox', 'tortoise', 'lizard', 'penguin'
]

nums = ['1', '2', '3', '4', '5', '6']

def generate_random_word_name():
    adj = random.choice(adjectives)
    noun = random.choice(nouns)
    number = random.choice(nums)
    return f'{adj}_{noun}_{number}'


def main():
	opts = TrainOptions().parse()
	expr_name = generate_random_word_name()
	opts.exp_dir = "./Results/"
	opts.exp_dir += expr_name+"/"
	wandb.init(name=expr_name ,config=opts, project='Mamba2_Gan_Inv', dir='./')
	os.makedirs(wandb.config.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = Coach(wandb.config)
	coach.train_jhon()


if __name__ == '__main__':
	main()
