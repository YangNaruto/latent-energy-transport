import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from torch import optim, autograd
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets


class Classifier(nn.Module):
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
			mlp.append(nn.Linear(n_hidden, 2))

		self.mlp = nn.Sequential(*mlp)


	def forward(self, x):
		return self.mlp(x)


class NPDataset(data.Dataset):
	def __init__(self, data,  transform=None):
		self.data = torch.from_numpy(data).float()
		self.transform = transform

	def __getitem__(self, index):
		x = self.data[index]

		if self.transform:
			x = self.transform(x)

		return x

	def __len__(self):
		return len(self.data)

model = Classifier(latent_dim=32, n_layer=2, n_hidden=128).cuda()
opt = optim.SGD(params=model.parameters(), lr=0.5)
src = np.load('female.npy')
tgt = np.load('male.npy')

test_data = np.load('female_gen.npy')
# sz = np.shape(src)[0]
# sample = np.concatenate((src, tgt), axis=0)
# label = np.concatenate((np.zeros(sz), np.ones(sz)))
# red = label == 0
# green = label == 1
# tsne = manifold.TSNE(n_components=2, init='random',
# 					 random_state=0, perplexity=100, n_iter=500)
# Y = tsne.fit_transform(sample)
#
# (fig, subplots) = plt.subplots(1, 2, figsize=(10, 5))
#
# ax = subplots[0]
# ax.scatter(Y[red, 0], Y[red, 1], c="r", s=5)
# ax.scatter(Y[green, 0], Y[green, 1], c="g", s=5)
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# ax.axis('tight')
#
# param = tsne.get_params()
# tsne= tsne.set_params(param)
#
#
#
#
#
# plt.savefig('tsne.png', dpi=300)
# plt.show()

batch_size = 64

src_loader = data.DataLoader(
	src,
	batch_size=batch_size,
	shuffle=True,
	num_workers=0,
	pin_memory=torch.cuda.is_available(),
	drop_last=True
)

tgt_loader = data.DataLoader(
	tgt,
	batch_size=batch_size,
	shuffle=True,
	num_workers=0,
	pin_memory=torch.cuda.is_available(),
	drop_last=True
)


src_iter = iter(src_loader)
tgt_iter = iter(tgt_loader)
used_sample = 0
iterations = -1
cross_ent = nn.CrossEntropyLoss(reduction='mean')
while used_sample < 10000000:
	iterations += 1
	model.zero_grad()
	opt.zero_grad()

	try:
		source_latent, target_latent = next(src_iter).to('cuda'), next(tgt_iter).to('cuda')
	except (OSError, StopIteration):
		src_iter = iter(src_loader)
		tgt_iter = iter(tgt_loader)
		source_latent, target_latent = next(src_iter).to('cuda'), next(tgt_iter).to('cuda')

	index = torch.randperm(batch_size * 2).long()
	source_label, target_label = torch.zeros_like(source_latent)[:, 0].long(), torch.ones_like(target_latent)[:, 0].long()
	labels = torch.cat((source_label, target_label), dim=0)[index]
	data = torch.cat((source_latent, target_latent), dim=0)[index]

	pred = model(data)
	loss = cross_ent(pred, labels)
	loss.backward()
	opt.step()

	correct = 0
	total = 0
	if iterations % 10 == 0:
		with torch.no_grad():
			test_sample = torch.from_numpy(test_data).cuda()
			test_labels = torch.zeros(1000).long().cuda()
			for step in range(10):
				pred = model(test_sample[step:step+1]).squeeze(0)
				_, predicted = torch.max(pred.data, 1)
				total += test_labels.size(0)
				correct += (predicted == test_labels).sum().item()

				print('Accuracy in Step %d: %d %%' % (step,
						100 * correct / total))


