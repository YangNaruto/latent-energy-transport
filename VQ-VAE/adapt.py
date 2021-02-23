import argparse
import os
from PIL import Image
import shutil
from torch.utils.data import DataLoader
from torch import optim, autograd
from torchvision import transforms, utils
import torchvision as tv
import torch.nn as nn
import torch
from torch.nn import functional as F
from collections import OrderedDict
import sys
import warnings
import torch.backends.cudnn as cudnn
import random
from glob import glob
import torch.utils.data as data
import numpy as np

from vqvae import VQVAE
from ulib.utils import read_single_image
import distributed as dist

from submit import _create_run_dir_local, _copy_dir
from logger import Logger
# from ebm import EBM
from model.stylegan1 import EBM


IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
	'.tif', '.TIF', '.tiff', '.TIFF',
]
policy = 'color,translation,cutout'

def ema(model1, model2, decay=0.999):
	par1 = dict(model1.named_parameters())
	par2 = dict(model2.named_parameters())

	for k in par1.keys():
		par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir

	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			if is_image_file(fname):
				path = os.path.join(root, fname)
				images.append(path)
	return images[:min(max_dataset_size, len(images))]


def default_loader(path):
	return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

	def __init__(self, root, transform=None, return_paths=False,
				 loader=default_loader):
		imgs = make_dataset(root)
		if len(imgs) == 0:
			raise (RuntimeError("Found 0 images in: " + root + "\n"
															   "Supported image extensions are: " + ",".join(
				IMG_EXTENSIONS)))

		self.root = root
		self.imgs = imgs
		self.transform = transform
		self.return_paths = return_paths
		self.loader = loader

	def __getitem__(self, index):
		path = self.imgs[index]
		img = self.loader(path)
		if self.transform is not None:
			img = self.transform(img)
		if self.return_paths:
			return img, path
		else:
			return img

	def __len__(self):
		return len(self.imgs)


def requires_grad(model, flag=True):
	for p in model.parameters():
		p.requires_grad = flag


def load_model(args, checkpoint, device):
	ckpt = torch.load(checkpoint)

	model = VQVAE(embed_dim=args.embed_dim, n_embed=args.n_embed, noise=args.noise)

	new_state_dict = OrderedDict()
	for k, v in ckpt.items():
		name = k.replace("module.", "")  # remove 'module.' of dataparallel
		new_state_dict[name] = v

	model.load_state_dict(new_state_dict)
	model = model.to(device)

	return model

def g_nonsaturating_loss(fake_pred):
	loss = F.softplus(-fake_pred).mean()
	return loss

def d_logistic_loss(real_pred, fake_pred):
	real_loss = F.softplus(-real_pred)
	fake_loss = F.softplus(fake_pred)
	return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
	grad_real, = autograd.grad(
		outputs=real_pred.sum(), inputs=real_img, create_graph=True
	)
	grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

	return grad_penalty


def langvin_sampler(model, x, langevin_steps=20, lr=1.0, sigma=0e-2, step=4, gamma=0.):
	x = x.clone().detach()
	x.requires_grad_(True)
	sgd = optim.SGD([x], lr=lr)
	for k in range(langevin_steps):
		model.zero_grad()
		sgd.zero_grad()
		energy = model(x).sum()
		# energy += -gamma * energy**2
		(-energy).backward()
		sgd.step()
	# f_prime = torch.autograd.grad(energy, [x], retain_graph=True)[0]
	# x.data += lr * f_prime + sigma * torch.randn_like(x)

	return x.clone().detach()

def dec_langvin_sampler(model, x, y, langevin_steps=10, lr=0.1, sigma=0e-2, step=4, gamma=0.):
	x = x.clone().detach()
	x.requires_grad_(True)
	sgd = optim.SGD([x], lr=lr)
	mse_loss = nn.MSELoss(reduction='mean')
	for k in range(langevin_steps):
		model.zero_grad()
		sgd.zero_grad()
		loss = mse_loss(x, y)
		# energy += -gamma * energy**2
		loss.backward()
		sgd.step()
	# f_prime = torch.autograd.grad(energy, [x], retain_graph=True)[0]
	# x.data += lr * f_prime + sigma * torch.randn_like(x)

	return x.clone().detach()

def check_folder(log_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	else:
		shutil.rmtree(log_dir)
		os.makedirs(log_dir)
	return log_dir


def test_image_folder(args, ae, ebm, refine_ebm=None, iteration=0, im_size=256, device='cuda'):
	data_root = os.path.join(args.data_root, args.dataset)
	source_files = list(glob(f'{data_root}/test/{args.source}/*.*'))
	root = os.path.join(args.run_dir, '{:06d}'.format(iteration))
	check_folder(root)
	requires_grad(ebm, False)
	for i, file in enumerate(source_files):
		if i == 1000:
			break
		if is_image_file(file):
			img_name = file.split("/")[-1]
			image = read_single_image(file, im_size=im_size, resize=True).to(device)
			latent = ae.latent(image)

			latent_q = langvin_sampler(ebm, latent, langevin_steps=args.langevin_step, lr=args.langevin_lr)
			image_t = ae.dec(latent_q)

			image_pair = torch.cat((image, image_t), dim=0)
			if args.refine:
				image_refined = langvin_sampler(refine_ebm, image_t.clone().detach())
				image_pair = torch.cat((image_pair, image_refined), dim=0)
			tv.utils.save_image(image_pair, os.path.join(root, img_name), padding=0, normalize=True, range=(-1, 1),
								nrow=1)


def main(args):
	ckpt = args.ae_ckpt
	device = "cuda"
	batch_size = args.batch_size
	args.distributed = dist.get_world_size() > 1

	# Preprocess data lodaer
	transform = transforms.Compose(
		[
			transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
			transforms.RandomHorizontalFlip(0.5),
			# transforms.Resize(256),
			# transforms.CenterCrop(256),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)

	data_root = os.path.join(args.data_root, args.dataset)

	source_dataset = ImageFolder(os.path.join(data_root, 'train/' + args.source), transform=transform)
	source_sampler = dist.data_sampler(source_dataset, shuffle=True, distributed=args.distributed)
	source_loader = DataLoader(
		source_dataset, batch_size=batch_size, sampler=source_sampler, num_workers=1, drop_last=True
	)
	target_dataset = ImageFolder(os.path.join(data_root, 'train/' + args.target), transform=transform)
	target_sampler = dist.data_sampler(target_dataset, shuffle=True, distributed=args.distributed)
	target_loader = DataLoader(
		target_dataset, batch_size=batch_size, sampler=target_sampler, num_workers=1, drop_last=True
	)
	source_iter = iter(source_loader)
	target_iter = iter(target_loader)

	# Define models
	ae = load_model(args, ckpt, device='cuda')
	ae.eval()

	latent_ebm = EBM(size=64, channel_multiplier=args.channel_mul, input_channel=args.embed_dim*2,
					 add_attention=args.attention,
					 spectral=args.sn, cam=args.cam).to(device)
	latent_ema = EBM(size=64, channel_multiplier=args.channel_mul, input_channel=args.embed_dim*2, add_attention=args.attention,
					 spectral=args.sn, cam=args.cam).to(device)
	ema(latent_ema, latent_ebm, decay=0.)
	latent_optimizer = optim.Adam(latent_ebm.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

	ae_param = sum(p.numel() for p in ae.parameters())
	ebm_param = sum(p.numel() for p in latent_ebm.parameters())
	print(ae_param, ebm_param)
	refine_ebm = None
	if args.refine:
		refine_ebm = EBM(size=256, channel_multiplier=1, input_channel=3, add_attention=args.attention,
						 spectral=args.sn, cam=args.cam).to(device)
		refine_optimizer = optim.Adam(refine_ebm.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

	used_sample = 0
	iterations = -1
	nrow = min(batch_size, 4)



	while used_sample < args.n_samples:
		iterations += 1
		latent_ebm.zero_grad()
		latent_optimizer.zero_grad()

		try:
			source_img, target_img = next(source_iter).to(device), next(target_iter).to(device)
		except (OSError, StopIteration):
			source_iter = iter(source_loader)
			target_iter = iter(target_loader)
			source_img, target_img = next(source_iter).to(device), next(target_iter).to(device)

		source_latent, target_latent = ae.latent(source_img), ae.latent(target_img)

		requires_grad(latent_ebm, False)
		source_latent_q = langvin_sampler(latent_ebm, source_latent.clone().detach(),
										  langevin_steps=args.langevin_step, lr=args.langevin_lr)

		requires_grad(latent_ebm, True)
		source_energy = latent_ebm(source_latent_q)
		target_energy = latent_ebm(target_latent)
		loss = -(target_energy - source_energy).mean()
		if args.l2:
			loss += (target_energy**2 + source_energy**2).mean()

		if abs(loss.item() > 100000) or loss.item() == np.nan:
			print('Diverge~~~~')
			sys.exit()

		loss.backward()
		latent_optimizer.step()

		ema(latent_ema, latent_ebm, decay=0.999)
		if args.refine:
			refine_ebm.zero_grad()
			refine_optimizer.zero_grad()
			requires_grad(refine_ebm, False)
			dec_q = ae.dec(source_latent_q)
			deq_qq = langvin_sampler(refine_ebm, dec_q.clone().detach())

			requires_grad(refine_ebm, True)
			source_energy = refine_ebm(deq_qq)
			target_energy = refine_ebm(target_img)
			refine_loss = -(target_energy - source_energy).mean()

			refine_loss.backward()
			refine_optimizer.step()

		used_sample += batch_size

		if iterations % 5000 == 0 and iterations != 0:
			test_image_folder(args, ae=ae, ebm=latent_ema, refine_ebm=refine_ebm, iteration=iterations, device=device)
			torch.save(latent_ebm.state_dict(), f"{args.run_dir}/ebm_{str(iterations).zfill(6)}.pt")

		if iterations % 500 == 0:
			print(f'Iter: {iterations:06}, Loss: {loss:6.3f}')

			latents = langvin_sampler(latent_ema, source_latent[:nrow].clone().detach(),
									  langevin_steps=args.langevin_step, lr=args.langevin_lr)
			with torch.no_grad():
				# latents = torch.cat((source_latent[:2], source_latent_q[:2]), dim=0)

				# latents = source_latent_q[:n_samples]
				out = ae.dec(latents)
				if args.refine:
					out = torch.cat((out, deq_qq[:nrow]), dim=0)
				out = torch.cat((source_img[:nrow], out), dim=0)
				utils.save_image(
					out,
					f"{args.run_dir}/{str(iterations).zfill(6)}.png",
					nrow=nrow,
					normalize=True,
					range=(-1, 1),
					padding=0
				)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--log_path", type=str, default='results')
	parser.add_argument("--n_samples", type=int, default=3_000_000)
	parser.add_argument("--n_gpu", type=int, default=1)
	parser.add_argument("--data_root", type=str)
	parser.add_argument("--ae_ckpt", type=str, default=None)

	# EBM Optimize
	parser.add_argument("--beta1", type=float, default=0.5)
	parser.add_argument("--beta2", type=float, default=0.999)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--lr", type=float, default=0.0025)

	parser.add_argument("--seed", type=int, help="seed number")

	parser.add_argument("--l2", action="store_true")
	parser.add_argument("--refine", action='store_true')
	parser.add_argument("--noise", action='store_true')

	# Langevin
	parser.add_argument("--langevin_step", type=int, default=20)
	parser.add_argument("--langevin_lr", type=float, default=1.0)

	# Architecture
	parser.add_argument("--attention", action='store_true', help='if use attention')
	parser.add_argument("--cam", action='store_true', help='if use cam')
	parser.add_argument("--sn", action='store_true', help='if use spectral norm')
	parser.add_argument("--blur", action='store_true', help='if add blur')
	parser.add_argument("--embed_dim", type=int, default=128, help="latent dimension (depth)")
	parser.add_argument("--n_embed", type=int, default=512, help="number of embeddings in the codebook")
	parser.add_argument("--channel_mul", type=int, default=2, help="channel multipliers")

	# Dataset
	parser.add_argument("--dataset", type=str, default='afhq')
	parser.add_argument("--source", type=str, default='cat')
	parser.add_argument("--target", type=str, default='dog')

	parser.add_argument("--suffix", type=str)

	# Data augmentation

	args = parser.parse_args()
	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	if args.ae_ckpt is None:
		args.ae_ckpt = os.path.join(os.path.join("./checkpoint", args.dataset), "/vqvae.pt")
	assert os.path.isfile(args.ae_ckpt), "input a valid ckpt"

	prefix = args.dataset + "-" + args.source + '2' + args.target
	save_path = os.path.join(args.log_path, prefix)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	suffix = "-".join([item for item in [
		"ls%d" % (args.langevin_step),
		"llr%.4f" % (args.langevin_lr),
		"lr%.4f" % (args.lr),
		"embed%d" % (args.embed_dim),
		"nembed%d" % (args.n_embed),
		"attn" if args.attention else None,
		"noise" if args.noise else None,
		"cam" if args.cam else None,
		"sn" if args.sn else None,
		"l2" if args.l2 else None,
		"chm%d" % args.channel_mul,
		"beta%.1f_%.3f" % (args.beta1, args.beta2),
		# "ada" if args.ada else None,
		"blur" if args.blur else None,
		"%s"%(args.suffix) if args.suffix else None
	] if item is not None])
	args.run_dir = _create_run_dir_local(save_path, suffix)
	_copy_dir(['adapt.py', 'ebm.py', 'model'], args.run_dir)



	sys.stdout = Logger(os.path.join(args.run_dir, 'log.txt'))
	print(args)
	main(args)
