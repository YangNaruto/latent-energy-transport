import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import torch.optim as optim
import seaborn as sns
sns.set()
seed = 5
np.random.seed(seed=seed)
torch.manual_seed(seed)
plt.rcParams.update({'font.size': 20})


def create_pie(n_samples=10000):
	rho = np.sqrt(np.random.uniform(0, 1, (n_samples, 1)))
	phi = np.random.uniform(0, 2 * np.pi, (n_samples, 1))

	x = rho * np.cos(phi)
	y = rho * np.sin(phi)

	return np.concatenate((x, y), axis=1)

def point_on_triangle(pt1, pt2, pt3):
	s, t = sorted([random.random(), random.random()])
	return (s * pt1[0] + (t-s)*pt2[0] + (1-t)*pt3[0],
			s * pt1[1] + (t-s)*pt2[1] + (1-t)*pt3[1])

def create_square(n_samples=10000):
	points = (np.random.rand(n_samples, 2) - 0.5) * 2 + np.array((0, 1))
	return points

def create_circle(origin=(0, 0), code="#c00000"):
	circle= plt.Circle(origin, radius=1, fill=True, alpha=0.6, color=code)
	return circle

def langvin_sampler(model, x, langevin_steps=8, lr=0.02, sigma=0e-2, seq=False):
	x = x.clone().detach()
	x.requires_grad_(True)
	sgd = optim.SGD([x], lr=lr)
	save = torch.zeros((x.shape[0], 8))
	for k in range(langevin_steps):
		x_org = x.clone().detach()
		model.zero_grad()
		sgd.zero_grad()
		energy = model(x).sum()

		(-energy).backward()
		sgd.step()
		save += torch.abs(x_org-x)
	if seq:
		return save.detach()
	else:
		return x.clone().detach()

class AutoEncoder(nn.Module):
	def __init__(self, neuron=10):
		super().__init__()

		self.encoder = nn.Linear(2, neuron)
		self.decoder = nn.Linear(neuron, 2)
		# self.encoder = nn.Sequential(*encoder)
		# self.decoder = nn.Sequential(*decoder)

	def forward(self, x):
		z = self.encoder(x)
		out = self.decoder(z)
		return z, out


class EBM(nn.Module):
	def __init__(self, neuron=10):
		super(EBM, self).__init__()

		self.layer = nn.ModuleList([nn.Linear(neuron, neuron),
									nn.GELU(),
									nn.Linear(neuron, 1)])
		self.layer = nn.Sequential(*self.layer)


	def forward(self, x):
		out = self.layer(x)
		return out

#----------------------------------
bs = 32
num_samples = 10000
source = 'pie_bottom'
log_dir = './logs'
if not os.path.exists(log_dir):
	os.mkdir(log_dir)

# Create two pies, bottom and top
pie_bottom = create_pie(n_samples=num_samples)
pie_top = create_pie() + np.array((0, 1))

#----------------------------------
ae = AutoEncoder(neuron=8)
ebm = EBM(neuron=8)
ae_opt = optim.SGD(params=ae.parameters(), lr=0.01)
ebm_opt = optim.SGD(params=ebm.parameters(), lr=0.02)

data = np.concatenate((pie_bottom, pie_top), axis=0)

mse = nn.MSELoss(reduction='mean')
print('Autoencoder pretraining...')
for i in range(5):
	np.random.permutation(data)
	x = torch.from_numpy(data[:num_samples*2]).float()
	_, output = ae(x)
	output = output.detach().numpy()
	plt.figure(figsize=(8, 8))
	plt.xlim(-2, 2)
	plt.ylim(-2, 2)
	plt.scatter(output[:, 0], output[:, 1], s=10)
	plt.savefig(f'{i:4d}.png', dpi=100)
	plt.close()

	loss = None
	for j in range(num_samples*2//bs):
		x_np = data[j*bs:(j+1)*bs]

		x = torch.from_numpy(x_np).float()
		_, out = ae(x)
		loss = mse(x,  out)

		ae_opt.zero_grad()
		loss.backward()
		ae_opt.step()
	print(f"Iteration: {i}, AE Loss: {loss.item():.4f}")



print('EBM Training...')
ae.eval()

if source == 'pie_bottom':
	source_data = pie_bottom
	target_data = pie_top
else:
	source_data = pie_top
	target_data = pie_bottom

fixed_src = torch.from_numpy(source_data[:num_samples]).float()
z_src, _ = ae(fixed_src)
target_t = ae.decoder(z_src).detach().numpy()
plt.figure(figsize=(8, 8))
plt.xlim(-2, 2)
plt.ylim(-2, 2)
circle = create_circle()
plt.gcf().gca().add_artist(circle, )
circle = create_circle(origin=(0, 1), code='#0070c0')
plt.gcf().gca().add_artist(circle, )

# t1 = plt.Polygon(pts, color='#0070c0', alpha=0.6)
# plt.gca().add_patch(t1)
# plt.scatter(target_t[:, 0], target_t[:, 1], s=10, c="k")
plt.axis('off')
plt.savefig(f'{log_dir}/shape.png', dpi=300)
plt.close()

for i in range(50):
	np.random.permutation(source_data)
	np.random.permutation(target_data)
	z_src, _ = ae(fixed_src)
	z_src_q = langvin_sampler(ebm, z_src)
	target_t = ae.decoder(z_src_q).detach().numpy()
	plt.figure(figsize=(8, 8))
	plt.xlim(-2, 2)
	plt.ylim(-2, 2)
	circle = create_circle()
	plt.gcf().gca().add_artist(circle, )
	circle = create_circle(origin=(0, 1),code='#0070c0')
	plt.gcf().gca().add_artist(circle, )
	# t1 = plt.Polygon(pts, color="#0070c0", alpha=0.6)
	# plt.gca().add_patch(t1)
	plt.scatter(target_t[:, 0], target_t[:, 1], s=10, c="k")
	plt.axis('off')
	plt.savefig(f'{log_dir}/seq_{i:03d}.png', dpi=300)
	plt.close()

	loss = 0.
	for j in range(num_samples // bs):
		source_in = torch.from_numpy(source_data[j*bs:(j+1)*bs]).float()
		target_in = torch.from_numpy(target_data[j*bs:(j+1)*bs]).float()
		z_src, _ = ae(source_in)
		z_tgt, _ = ae(target_in)

		z_src_q = langvin_sampler(ebm, z_src)
		source_energy = ebm(z_src_q)
		target_energy = ebm(z_tgt)
		loss = -(target_energy - source_energy).mean()

		ebm_opt.zero_grad()
		loss.backward()
		ebm_opt.step()
	print(f"Epoch: {i}, EBM Loss: {loss:.4f}")

# Visualize the latent
z,_ = ae(fixed_src)
z_change = langvin_sampler(ebm, z, seq=True).numpy()
plt.figure(figsize=(5, 5))
y_ticks = np.arange(0, z.shape[0], 200)

ax = sns.heatmap(z_change)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
# ax.yaxis.set_ticks_position('none')
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.ylabel('2000 Samples', fontdict={'size':16})
plt.xlabel('$z$', fontdict={'size':16})
plt.xticks(fontsize= 14 )
plt.savefig(f'{log_dir}/heatmap.png', dpi=300)
plt.close()
print(z_change)
