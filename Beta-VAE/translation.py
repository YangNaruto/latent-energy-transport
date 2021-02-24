import torch
import torch.nn as nn
from torch import optim, autograd
import os, argparse
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import shutil
import numpy as np
from model import BetaVAE_H, BetaVAE_B
from dummy import read_single_image
from submit import _create_run_dir_local, _copy_dir
import random
import torchvision as tv
from glob import glob


IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
	'.tif', '.TIF', '.tiff', '.TIFF',
]
transform = transforms.Compose(
	[
		transforms.Resize(64),
		transforms.CenterCrop(64),
		transforms.RandomHorizontalFlip(0.5),

		# transforms.CenterCrop(256),
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
	]
)

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

def langvin_sampler(model, x, langevin_steps=20, lr=1.0, sigma=0e-2, return_seq=False):
	x = x.clone().detach()
	x.requires_grad_(True)
	sgd = optim.SGD([x], lr=lr)

	sequence = torch.zeros_like(x).unsqueeze(0).repeat(langevin_steps, 1, 1)
	for k in range(langevin_steps):
		sequence[k] = x.data
		model.zero_grad()
		sgd.zero_grad()
		energy = model(x).sum()

		(-energy).backward()
		sgd.step()

	if return_seq:
		return sequence
	else:
		return x.clone().detach()

def check_folder(log_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	else:
		shutil.rmtree(log_dir)
		os.makedirs(log_dir)
	return log_dir

def requires_grad(model, flag=True):
	for p in model.parameters():
		p.requires_grad = flag

def test_image_folder(args, ae, ebm, run_dir, iteration=0, device='cuda'):
	data_root = os.path.join("../i2i/datasets", args.dataset)
	source_files = list(glob(f'{data_root}/test/{args.source}/*.*'))
	root = os.path.join(run_dir, '{:06d}'.format(iteration))
	check_folder(root)
	requires_grad(ebm, False)
	random.shuffle(source_files)
	latent_seqs = []
	for i, file in enumerate(source_files[:min(len(source_files), 1000)]):
		if is_image_file(file):
			img_name = file.split("/")[-1]
			image = read_single_image(file, im_size=64, resize=True).to(device)
			latent = ae._encode(image)[:, :args.z_dim]

			latent_seq = langvin_sampler(ebm, latent.clone().detach(), langevin_steps=args.langevin_step,
									   lr=args.langevin_lr, return_seq=True)
			# np.save(os.path.join(root, img_name[:-4] + '.npy'), latent_seq.cpu().numpy())
			latent_seqs.append(latent_seq.cpu().numpy())
			img_seq = ae._decode(latent_seq)
			img_seq = torch.cat((image, img_seq), dim=0)
			tv.utils.save_image(img_seq, os.path.join(root, img_name), padding=0, normalize=True, range=(-1., 1.),
								nrow=img_seq.shape[0])
	np.save(os.path.join(root, f'{args.source}_gen.npy'), np.concatenate(latent_seqs, axis=1))


def train(args, logger=None):
	device = 'cuda'
	# Create save path
	prefix = args.dataset + "-" + args.source + '2' + args.target
	save_path = os.path.join("results", prefix)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	suffix = "-".join([item for item in [
		"ls%d" % (args.langevin_step),
		"llr%.2f" % (args.langevin_lr),
		"lr%.4f" % (args.lr),
		"h%d" % (args.hidden),
		"layer%d" % (args.layer),
	] if item is not None])
	run_dir = _create_run_dir_local(save_path, suffix)
	_copy_dir(['translation.py'], run_dir)


	if os.path.isfile(args.ckpt):
		checkpoint = torch.load(args.ckpt)
	else:
		print('No valid ckpt')
	ae = BetaVAE_H(args.z_dim, 3).cuda()
	ae.load_state_dict(checkpoint['model_states']['net'])
	ae.eval()

	batch_size = 128
	data_root = os.path.join('../i2i/datasets', args.dataset)
	print(data_root)
	source_dataset = ImageFolder(os.path.join(data_root, 'train/' + args.source), transform=transform)

	source_loader = DataLoader(
		source_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
	)
	target_dataset = ImageFolder(os.path.join(data_root, 'train/' + args.target), transform=transform)

	target_loader = DataLoader(
		target_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
	)
	source_iter = iter(source_loader)
	target_iter = iter(target_loader)


	latent_ebm = LatentEBM(latent_dim=args.z_dim, n_layer=args.layer, n_hidden=args.hidden).cuda()
	latent_ema = LatentEBM(latent_dim=args.z_dim, n_layer=args.layer, n_hidden=args.hidden).cuda()

	ema(latent_ema, latent_ebm, decay=0.)

	latent_optimizer = optim.SGD(latent_ebm.parameters(), lr=args.lr)


	used_sample = 0
	iterations = -1
	nrow = min(batch_size, 8)


	for key, loader in {args.target: target_loader, args.source: source_loader}.items():
		latents = []
		used_sample = 0
		for i, img in enumerate(loader):
			latent = ae._encode(img.cuda())[:, :args.z_dim]
			latents.append(latent.detach().cpu().numpy())
			used_sample += batch_size
			if used_sample > 10000:
				break
		latents = np.concatenate(latents, axis=0)
		np.save(f"{key}.npy", latents)


	while used_sample < 10000000:
		iterations += 1
		latent_ebm.zero_grad()
		latent_optimizer.zero_grad()

		try:
			source_img, target_img = next(source_iter).to(device), next(target_iter).to(device)
		except (OSError, StopIteration):
			source_iter = iter(source_loader)
			target_iter = iter(target_loader)
			source_img, target_img = next(source_iter).to(device), next(target_iter).to(device)

		# source_latent, target_latent = ae.reparameterize(source_img), ae.reparameterize(target_img)
		source_latent, target_latent = ae._encode(source_img)[:, :args.z_dim], ae._encode(target_img)[:, :args.z_dim]

		requires_grad(latent_ebm, False)
		source_latent_q = langvin_sampler(latent_ebm, source_latent.clone().detach(),
										  langevin_steps=args.langevin_step, lr=args.langevin_lr,)

		requires_grad(latent_ebm, True)
		source_energy = latent_ebm(source_latent_q)
		target_energy = latent_ebm(target_latent)
		loss = -(target_energy - source_energy).mean()

		if abs(loss.item() > 10000):
			break
		loss.backward()
		latent_optimizer.step()

		ema(latent_ema, latent_ebm, decay=0.99)

		used_sample += batch_size
		#
		if iterations % 1000 == 0:
			test_image_folder(args, ae=ae, ebm=latent_ema, run_dir=run_dir, iteration=iterations, device=device)
			# test_representatives(cfg=cfg, ae=ae, ebm=latent_ema, run_dir=run_dir, iteration=iterations, device=device)
			torch.save(latent_ebm.state_dict(), f"{run_dir}/ebm_{str(iterations).zfill(6)}.pt")

		if iterations % 100 == 0:
			print(f'Iter: {iterations:06}, Loss: {loss:6.3f}')

			latents = langvin_sampler(latent_ema, source_latent[:nrow].clone().detach(),
									  langevin_steps=args.langevin_step, lr=args.langevin_lr,)


			with torch.no_grad():
				latents = torch.cat((source_latent[:nrow], latents), dim=0)

				out = ae._decode(latents)

				out = torch.cat((source_img[:nrow], out), dim=0)
				utils.save_image(
					out,
					f"{run_dir}/{str(iterations).zfill(6)}.png",
					nrow=nrow,
					normalize=True,
					padding=0,
					range=(-1, 1)
				)




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, default='celeba')
	parser.add_argument("--source", type=str, default='female')
	parser.add_argument("--target", type=str, default='male')
	parser.add_argument("--ckpt", type=str)

	# Langevin
	parser.add_argument("--langevin_step", type=int, default=10)
	parser.add_argument("--langevin_lr", type=float, default=0.1)
	parser.add_argument("--lr", type=float, default=0.1)
	parser.add_argument("--layer", type=int, default=1)
	parser.add_argument("--hidden", type=int, default=64)
	parser.add_argument("--z_dim", type=int, default=32)

	args = parser.parse_args()

	train(args)

