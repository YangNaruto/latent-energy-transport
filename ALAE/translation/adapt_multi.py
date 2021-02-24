# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import shutil
import torch.utils.data
import torchvision as tv
from glob import glob
from torchvision.utils import save_image
import random
import sys
import numpy as np
import os
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch import optim, autograd
from torchvision import transforms, utils
from itertools import permutations
from imbalanced import ImbalancedDatasetSampler
from dummy import read_single_image
from submit import _create_run_dir_local, _copy_dir
from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
from ebm import LatentEBM, MulLatentEBM
from logger import Logger
import distributed as dist
import tqdm
from PIL import Image
from sgld import SGLD

lreq.use_implicit_lreq.set(True)

IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
	'.tif', '.TIF', '.tiff', '.TIFF',
]


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

def langvin_sampler(model, x, layer_label=None, langevin_steps=20, lr=1.0, sigma=0e-2, return_seq=False):
	x = x.clone().detach()
	x.requires_grad_(True)
	# sgd = optim.SGD([x], lr=lr)
	sgd = SGLD([x], lr=lr, std_dev=sigma)
	sequence = torch.zeros_like(x).unsqueeze(0).repeat(langevin_steps, 1, 1)
	for k in range(langevin_steps):
		sequence[k] = x.data
		model.zero_grad()
		sgd.zero_grad()
		energy = model(x, label=layer_label).sum()

		(-energy).backward()
		sgd.step()

	if return_seq:
		return sequence
	else:
		return x.clone().detach()

def load_ae(cfg, logger):
	torch.cuda.set_device(0)
	model = Model(
		startf=cfg.MODEL.START_CHANNEL_COUNT,
		layer_count=cfg.MODEL.LAYER_COUNT,
		maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
		latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
		truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
		truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
		mapping_layers=cfg.MODEL.MAPPING_LAYERS,
		channels=cfg.MODEL.CHANNELS,
		generator=cfg.MODEL.GENERATOR,
		encoder=cfg.MODEL.ENCODER)
	model.cuda()
	model.eval()
	model.requires_grad_(False)


	decoder = model.decoder
	encoder = model.encoder
	mapping_tl = model.mapping_tl
	mapping_fl = model.mapping_fl
	dlatent_avg = model.dlatent_avg

	logger.info("Trainable parameters generator:")
	count_parameters(decoder)

	logger.info("Trainable parameters discriminator:")
	count_parameters(encoder)

	arguments = dict()
	arguments["iteration"] = 0

	model_dict = {
		'discriminator_s': encoder,
		'generator_s': decoder,
		'mapping_tl_s': mapping_tl,
		'mapping_fl_s': mapping_fl,
		'dlatent_avg': dlatent_avg
	}

	checkpointer = Checkpointer(cfg,
								model_dict,
								{},
								logger=logger,
								save=False)

	extra_checkpoint_data = checkpointer.load()

	model.eval()

	layer_count = cfg.MODEL.LAYER_COUNT

	path = cfg.DATASET.SAMPLES_PATH
	im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)
	return model

def check_folder(log_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	else:
		shutil.rmtree(log_dir)
		os.makedirs(log_dir)
	return log_dir

def encode(model, x, cfg):

	Z, _ = model.encode(x, cfg.MODEL.LAYER_COUNT - 1, 1)
	# Z = Z.repeat(1, ae.mapping_fl.num_layers, 1)
	return Z.squeeze(1)

#
def decode(model, x, cfg):
	layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
	ones = torch.ones(layer_idx.shape, dtype=torch.float32)
	coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
	# x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
	return model.decoder(x, cfg.MODEL.LAYER_COUNT - 1, 1, noise=True)

def generate_recon(cfg, ae, ebm, run_dir, iteration=0, device='cuda', n_sample=1):
	data_root = os.path.join(cfg.DATA.ROOT, cfg.DATA.NAME)
	source_files = list(glob(f'{data_root}/test/female/*.*'))
	root = os.path.join(run_dir, '{:06d}'.format(iteration))
	check_folder(os.path.join(run_dir, 'recon'))
	requires_grad(ebm, False)
	for i, file in enumerate(source_files):
		if is_image_file(file):
			img_name = file.split("/")[-1]
			image = read_single_image(file, im_size=2**cfg.DATASET.MAX_RESOLUTION_LEVEL, resize=True).to(device)
			latent = encode(ae, image, cfg)
			latents = latent.repeat(1, ae.mapping_fl.num_layers, 1)
			image_t = decode(ae, latents, cfg)
			tv.utils.save_image(image_t, os.path.join(os.path.join(run_dir, 'recon'), img_name), padding=0, normalize=True, range=(-1., 1.),
								nrow=1)
	print('Reconstruction done !!')

def test_image_folder(cfg, ae, ebm, run_dir, iteration=0, device='cuda', n_sample=3):
	data_root = os.path.join(cfg.DATA.ROOT, cfg.DATA.NAME)
	source_files = list(glob(f'{data_root}/representatives/{cfg.DATA.SOURCE}/*.*'))
	root = os.path.join(run_dir, '{:06d}'.format(iteration))
	check_folder(root)
	requires_grad(ebm, False)
	random.shuffle(source_files)
	for i, file in enumerate(source_files):
		# if 2**cfg.DATASET.MAX_RESOLUTION_LEVEL==1024 and i > 100:
		# 	break
		if is_image_file(file):
			img_name = file.split("/")[-1]
			image = read_single_image(file, im_size=2**cfg.DATASET.MAX_RESOLUTION_LEVEL, resize=True).to(device)
			latent = encode(ae, image, cfg)

			latent_q = langvin_sampler(ebm, latent.clone().detach(), langevin_steps=cfg.LANGEVIN.STEP,
									   lr=cfg.LANGEVIN.LR)

			# latents = latent_q.unsqueeze(1).repeat(1, ae.mapping_fl.num_layers, 1)
			latents = torch.stack((latent, latent_q), dim=0)
			latents = latents.repeat(1, ae.mapping_fl.num_layers, 1)
			image_t = decode(ae, latents, cfg)
			image_pair = torch.cat((image, image_t), dim=0)

			for _ in range(n_sample-1):
				latent_q = langvin_sampler(ebm, latent.clone().detach(), langevin_steps=cfg.LANGEVIN.STEP,
										   lr=cfg.LANGEVIN.LR)
				image_t = decode(ae, latent_q.repeat(1, ae.mapping_fl.num_layers, 1), cfg)
				image_pair = torch.cat((image_pair, image_t), dim=0)

			tv.utils.save_image(image_pair, os.path.join(root, img_name), padding=0, normalize=True, range=(-1., 1.),
								nrow=1)

def test_representatives(cfg, ae, ebm, run_dir, iteration=0, device='cuda'):
	data_root = os.path.join(cfg.DATA.ROOT, cfg.DATA.NAME)
	source_files = list(glob(f'{data_root}/test/{cfg.DATA.SOURCE}/*.*'))
	root = os.path.join(run_dir, '{:06d}'.format(iteration))
	check_folder(root)
	requires_grad(ebm, False)
	random.shuffle(source_files)
	for i, file in enumerate(source_files):

		if is_image_file(file):
			img_name = file.split("/")[-1]
			image = read_single_image(file, im_size=2**cfg.DATASET.MAX_RESOLUTION_LEVEL, resize=True).to(device)
			latent = encode(ae, image, cfg)

			latent_seq = langvin_sampler(ebm, latent.clone().detach(), langevin_steps=cfg.LANGEVIN.STEP,
									   lr=cfg.LANGEVIN.LR, return_seq=True)
			np.save(os.path.join(root, img_name[:-4]+'.npy'), latent_seq.cpu().numpy())
			latent_seq = latent_seq[::2].repeat(1, ae.mapping_fl.num_layers, 1)
			# for j in range(0, cfg.LANGEVIN.STEP+1, 2):
			# 	cur_latent = latent_seq[j].repeat(1, ae.mapping_fl.num_layers, 1)
			# 	cur_image = decode(ae, cur_latent, cfg)
			img_seq = decode(ae, latent_seq, cfg)
			tv.utils.save_image(img_seq, os.path.join(root, img_name), padding=0, normalize=True, range=(-1., 1.),
								nrow=img_seq.shape[0])


def train(cfg, logger):
	# Create save path
	prefix = cfg.DATA.NAME + "-all"
	save_path = os.path.join("results", prefix)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	suffix = "-".join([item for item in [
		"ls%d" % (cfg.LANGEVIN.STEP),
		"llr%.2f" % (cfg.LANGEVIN.LR),
		"lr%.4f" % (cfg.EBM.LR),
		"h%d" % (cfg.EBM.HIDDEN),
		"layer%d" % (cfg.EBM.LAYER),
		"opt%s" % (cfg.EBM.OPT),
	] if item is not None])
	run_dir = _create_run_dir_local(save_path, suffix)
	_copy_dir(['translation'], run_dir)
	sys.stdout = Logger(os.path.join(run_dir, 'log.txt'))

	ae = load_ae(cfg, logger)

	device = 'cuda'
	transform = transforms.Compose(
		[
			transforms.RandomResizedCrop(2**cfg.DATASET.MAX_RESOLUTION_LEVEL, scale=[0.8, 1.0], ratio=[0.9, 1.1]),
			transforms.RandomHorizontalFlip(0.5),
			# transforms.Resize(256),
			# transforms.CenterCrop(256),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)

	data_root = os.path.join(cfg.DATA.ROOT, cfg.DATA.NAME)
	print(data_root)
	all_dataset = tv.datasets.ImageFolder(os.path.join(data_root, 'train/'), transform=transform)
	# all_sampler = dist.data_sampler(all_dataset, shuffle=True, distributed=False)
	all_loader = DataLoader(
		all_dataset, batch_size=cfg.DATA.BATCH, sampler=ImbalancedDatasetSampler(all_dataset), num_workers=0, drop_last=True
	)

	latent_ebm = MulLatentEBM(latent_dim=512, n_layer=cfg.EBM.LAYER, n_hidden=cfg.EBM.HIDDEN).cuda()

	latent_ema = MulLatentEBM(latent_dim=512, n_layer=cfg.EBM.LAYER, n_hidden=cfg.EBM.HIDDEN).cuda()
	ema(latent_ema, latent_ebm, decay=0.)

	latent_optimizer = optim.SGD(latent_ebm.parameters(), lr=cfg.EBM.LR)
	if cfg.EBM.OPT == 'adam':
		latent_optimizer = optim.Adam(latent_ebm.parameters(), lr=cfg.EBM.LR)

	layer_count = cfg.MODEL.LAYER_COUNT
	used_sample = 0
	iterations = -1
	nrow = min(cfg.DATA.BATCH, 4)
	batch_size = cfg.DATA.BATCH

	ebm_param = sum(p.numel() for p in latent_ebm.parameters())
	ae_param = sum(p.numel() for p in ae.parameters())
	print(ebm_param, ae_param)
	n_classes = 3 if cfg.DATA.NAME == 'afhq' else 2
	choices = list(permutations(range(n_classes), 2))
	classifier_loss = torch.nn.CrossEntropyLoss(reduction='sum').cuda()
	for epoch in range(200):
		for step, (x, y) in enumerate(all_loader):
			if len(set(list(y.squeeze().cpu().numpy()))) == 1:
				continue

			iterations += 1
			latent_ebm.zero_grad()
			latent_optimizer.zero_grad()

			images, labels = x.to(device), y.to(device)

			latents = encode(ae, images, cfg).squeeze()
			sorted_labels, sorted_indices = torch.sort(labels)
			source_latents = latents[sorted_indices]
			source_images = images[sorted_indices]
			target_latents = torch.zeros_like(source_latents).clone().detach()
			target_labels = torch.zeros_like(sorted_labels).clone().detach()

			t = 0
			for l in range(n_classes):
				label_ind = torch.nonzero(sorted_labels!=l).squeeze()
				label_num = len(label_ind)
				rnd_ind = torch.randint(label_num, (batch_size-label_num,))
				target_latents[t:t+batch_size-label_num] = source_latents[label_ind[rnd_ind]]
				target_labels[t:t+batch_size-label_num] = sorted_labels[label_ind[rnd_ind]]
				t += (batch_size - label_num)

			layer_labels = torch.zeros_like(labels).clone().detach()
			for l in range(batch_size):
				ind = choices.index((sorted_labels[l].cpu().numpy(), target_labels[l].cpu().numpy()))
				layer_labels[l] = torch.tensor(ind).long().cuda()


			requires_grad(latent_ebm, False)
			source_latent_q = langvin_sampler(latent_ebm, source_latents.clone().detach(), layer_label=layer_labels,
											  langevin_steps=cfg.LANGEVIN.STEP, lr=cfg.LANGEVIN.LR,)

			requires_grad(latent_ebm, True)
			source_energy = latent_ebm(source_latent_q, label=None)
			target_energy = latent_ebm(target_latents, label=None)
			loss = -(target_energy - source_energy)

			source_pred = latent_ebm(source_latent_q, label=layer_labels)
			target_pred = latent_ebm(target_latents, label=layer_labels)
			ccf_loss = classifier_loss(source_pred, layer_labels) + classifier_loss(target_pred, layer_labels)
			loss = torch.mean(loss + ccf_loss)

			# if abs(loss.item() > 10000):
			# 	break
			loss.backward()
			latent_optimizer.step()

			ema(latent_ema, latent_ebm, decay=0.999)

			used_sample += batch_size
			# #
			# if iterations % 1000 == 0:
			# 	test_image_folder(cfg=cfg, ae=ae, ebm=latent_ema, run_dir=run_dir, iteration=iterations, device=device)
			# 	# test_representatives(cfg=cfg, ae=ae, ebm=latent_ema, run_dir=run_dir, iteration=iterations, device=device)
			# 	torch.save(latent_ebm.state_dict(), f"{run_dir}/ebm_{str(iterations).zfill(6)}.pt")

			rand_sel = np.arange(batch_size)
			rand_sel = np.random.permutation(rand_sel)
			rand_sel = rand_sel[:nrow]
			if iterations % 100 == 0:
				print(f'Iter: {iterations:06}, Loss: {loss:6.3f}')

				latents = langvin_sampler(latent_ema, source_latents[rand_sel].clone().detach(), layer_label=layer_labels[rand_sel],
										  langevin_steps=cfg.LANGEVIN.STEP, lr=cfg.LANGEVIN.LR)

				with torch.no_grad():
					latents = torch.cat((source_latents[rand_sel], latents))
					latents = latents.unsqueeze(1).repeat(1, ae.mapping_fl.num_layers, 1)

					out = decode(ae, latents, cfg)

					out = torch.cat((source_images[rand_sel], out), dim=0)
					utils.save_image(
						out,
						f"{run_dir}/{str(iterations).zfill(6)}.png",
						nrow=nrow,
						normalize=True,
						padding=0,
						range=(-1, 1),
					)




if __name__ == "__main__":
	gpu_count = 1
	run(train, get_cfg_defaults(), description='Image-Translation', default_config='configs/celeba-hq.yaml',
		world_size=gpu_count, write_log=False)
