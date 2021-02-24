import torch
import torch.nn as nn
import numpy as np

from itertools import permutations


class MulLatentEBM(nn.Module):
	def __init__(self, latent_dim=512, n_layer=4, n_hidden=2048, n_classes=2):
		super().__init__()

		mlp = nn.ModuleList()
		choices = list(permutations(range(n_classes), 2))
		num_choices = len(choices)

		self.energy_output = nn.Linear(n_hidden, 1)
		self.class_output = nn.Linear(n_hidden, num_choices)
		# self.class_output = nn.ModuleDict()
		# for i in range(num_choices):
		# 	self.class_output[str(i)] = nn.Linear(n_hidden, num_choices)

		if n_layer == 0:
			mlp.append(nn.Linear(latent_dim, 1))
		else:
			mlp.append(nn.Linear(latent_dim, n_hidden))

			for _ in range(n_layer-1):
				mlp.append(nn.LeakyReLU(0.2))
				mlp.append(nn.Linear(n_hidden, n_hidden))

			# mlp.append(nn.LeakyReLU(0.2))
			# mlp.append(nn.Linear(n_hidden, hi))

		self.last_layer = nn.Linear(n_hidden, 1)
		self.activate = nn.LeakyReLU(0.2)
		self.mlp = nn.Sequential(*mlp)
		self.embedding = nn.Embedding(num_choices, n_hidden)

	def forward(self, x, label=None):
		h = self.mlp(x)
		h = self.activate(h)
		logits = self.class_output(h).squeeze()

		if label is not None:
			return torch.gather(logits, 1, label[:, None])
		else:
			return logits.logsumexp(1)

		# out = self.last_layer(h)
		# out = out + torch.mean(self.embedding(label) * h , dim=1, keepdim=True)

		# return out, energy



class LatentEBM(nn.Module):
	def __init__(self, latent_dim=512, n_layer=4, n_hidden=2048):
		super().__init__()

		mlp = nn.ModuleList()
		if n_layer == 0:
			mlp.append(nn.Linear(latent_dim, 1))
		else:
			mlp.append(nn.Linear(latent_dim, n_hidden))

			for _ in range(n_layer-1):
				mlp.append(nn.LeakyReLU(0.2))
				mlp.append(nn.Linear(n_hidden, n_hidden))

			mlp.append(nn.LeakyReLU(0.2))
			mlp.append(nn.Linear(n_hidden, 1))

		self.mlp = nn.Sequential(*mlp)


	def forward(self, x):
		return self.mlp(x)

class NoiseInjection(nn.Module):
	def __init__(self):
		super().__init__()

		self.weight = nn.Parameter(torch.zeros(1))

	def forward(self, image, noise=None):
		if noise is None:
			# batch, _, height, width = image.shape
			# noise = image.new_empty(batch, 1, height, width).normal_()

			batch, num_neuron = image.shape
			noise = image.new_empty(batch, num_neuron).normal_()

		return image + self.weight * noise
