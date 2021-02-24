import os
import shutil
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as trans_fn
import torchvision.transforms as transforms
import random
import math
import matplotlib.pyplot as plt
import torchvision as tv


class EMA(object):
	def __init__(self, source, target, decay=0.9999, start_itr=0):
		self.source = source
		self.target = target
		self.decay = decay
		# Optional parameter indicating what iteration to start the decay at
		# Initialize target's params to be source's
		self.source_dict = self.source.state_dict()
		self.target_dict = self.target.state_dict()
		print('Initializing EMA parameters to be source parameters...')
		with torch.no_grad():
			for key in self.source_dict:
				self.target_dict[key].data.copy_(self.source_dict[key].data)
				# target_dict[key].data = source_dict[key].data # Doesn't work!

	def update(self, decay=0.0):

		with torch.no_grad():
			for key in self.source_dict:
				self.target_dict[key].data.copy_(self.target_dict[key].data * decay
				                                 + self.source_dict[key].data * (1 - decay))

def imread(filename, size, resize=True):
	"""
	Loads an image file into a (height, width, 3) uint8 ndarray.
	"""
	img = Image.open(filename)
	if resize:
		img = im_resize(img, size=size)
	return np.asarray(img, dtype=np.uint8)[..., :3]

def im_resize(img, size=256):
	img = trans_fn.resize(img, size, Image.LANCZOS)
	img = trans_fn.center_crop(img, size)

	return img

def plot_heatmap(data, fig_name):
	a2b = data
	bs, steps, ch, h, w = a2b.shape
	# print(f'steps: {steps}, bs: {bs}, size: {h, w}')

	residual = np.zeros((bs, ch, h, w))
	for step in range(steps - 1):
		residual += np.abs(a2b[:, step + 1] - a2b[:, step])

	residual = np.mean(residual, axis=1)

	n_rows = int(math.sqrt(bs))
	n_cols = int(math.sqrt(bs))

	residual = np.reshape(residual, (n_rows, n_cols, w, h))
	residual = np.transpose(residual, axes=(0, 2, 1, 3))
	residual = np.reshape(residual, (n_rows * w, n_cols * h))

	fig, ax = plt.subplots(1, 1)
	plt.axis('off')

	ax.set_xticklabels('')
	ax.set_yticklabels('')
	plt.imshow(residual)

	plt.savefig(fig_name, dpi=300)
	plt.close(fig)

def read_single_image(file, im_size, resize=True):
	image = imread(str(file), size=im_size, resize=resize).astype(np.float32)
	image = np.expand_dims(image, 0).transpose((0, 3, 1, 2))
	image = (image / 255 - 0.5) * 2
	image = torch.from_numpy(image).type(torch.FloatTensor)
	return image.contiguous()

def read_image_folder(files, batch_size, im_size, resize=True):
	# random.shuffle(files)

	images = np.array([imread(str(f), size=im_size, resize=resize).astype(np.float32)
	                   for f in files[:batch_size]])

	# Reshape to (n_images, 3, height, width)
	images = images.transpose((0, 3, 1, 2))
	images = (images / 255 - 0.5) * 2
	batch = torch.from_numpy(images).type(torch.FloatTensor)

	return batch

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def nearest_neighbor(sample, dataset, logdir, i, k=10, im_size=64):
	sample_num, length = sample.shape[0], sample.shape[2]
	assert len(sample.shape) == 3 and len(dataset.shape) == 3

	distance = torch.pow(sample - dataset, 2).mean(dim=-1)
	distance_idx = torch.argsort(distance, dim=1)

	# Select the k-nn
	panel_all = torch.zeros((sample_num, k+1, length), device=sample.device)
	distance_idx = distance_idx[:, :k]

	dataset = torch.squeeze(dataset, dim=0)
	panel_nn = dataset[distance_idx.reshape(-1)].reshape(sample_num, k, -1)

	panel_all[:, 1:] = panel_nn
	panel_all[:, :1] = sample
	panel_all = panel_all.reshape(sample_num*(k+1), 3, im_size, im_size)

	tv.utils.save_image(panel_all, '{}/sample_{:02d}.png'.format(logdir, i), padding=2, normalize=True,
	                    range=(-1., 1.),
	                    nrow=k+1)

def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	else:
		shutil.rmtree(path)
		os.makedirs(path)
