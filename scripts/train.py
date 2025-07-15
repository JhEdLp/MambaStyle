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


def main():
	opts = TrainOptions().parse()
	
	if 'ffhq' in opts.dataset_type:
		from training.coach import Coach
	else: 
		from training.coach_cars import Coach

	os.makedirs(opts.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = Coach(opts)
	coach.train_jhon()


if __name__ == '__main__':
	main()
