import torch
import torch.nn.functional as F


def aggregate_loss_dict(agg_loss_dict):
	mean_vals = {}
	for output in agg_loss_dict:
		for key in output:
			mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
	for key in mean_vals:
		if len(mean_vals[key]) > 0:
			mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
		else:
			print('{} has no value'.format(key))
			mean_vals[key] = 0
	return mean_vals


class ImageResizer:
    def __init__(self, size=(256, 256), mode='bilinear', antialias=True):
        self.size = size
        self.mode = mode
        self.antialias = antialias

    def interpolate(self, x):
        return F.interpolate(x, size=self.size, mode=self.mode, antialias=self.antialias)