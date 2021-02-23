import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm
from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist
from ulib.utils import create_dir
from ulib.data_utils import mixup_data, cut_mix
from ulib.logger import Logger
from ulib.non_leaking import augment


seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)


def train(epoch, loader, model, optimizer, scheduler, device, args):
	if dist.is_primary():
		loader = tqdm(loader)

	criterion = nn.MSELoss()

	latent_loss_weight = 0.25
	sample_size = 8
	avg_mse= 9999
	mse_sum = 0
	mse_n = 0

	for i, (img, label) in enumerate(loader):
		model.zero_grad()

		img = img.to(device)
		img_copy = img.clone().detach()
		lam_mixup, lam_cutmix = 1.0, 1.0
		rand_index = torch.arange(0, img.size()[0]).cuda()
		if args.mixup:
			img, rand_index, lam_mixup = mixup_data(x=img, spherical=args.spherical)

		if args.cutmix:
			img, rand_index, lam_cutmix = cut_mix(x=img, args=args)

		if args.ada:
			img = augment(img, p=args.ada_p)

		lam = lam_cutmix * lam_mixup
		out, latent_loss = model(img + args.input_noise * torch.randn_like(img), rand_index=None, lam=lam)

		recon_loss = criterion(out, img)
		latent_loss = latent_loss.mean()
		loss = recon_loss + latent_loss_weight * latent_loss
		loss.backward()

		if scheduler is not None:
			scheduler.step()
		optimizer.step()

		part_mse_sum = recon_loss.item() * img.shape[0]
		part_mse_n = img.shape[0]
		comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
		comm = dist.all_gather(comm)

		for part in comm:
			mse_sum += part["mse_sum"]
			mse_n += part["mse_n"]

		if dist.is_primary():
			lr = optimizer.param_groups[0]["lr"]
			avg_mse =  mse_sum / mse_n

			loader.set_description(
				(	f"epoch: {epoch + 1}/{args.total_epoch}; "
					f"mse: {recon_loss.item():.5f}; "
					f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
					f"lr: {lr:.5f}"
				)
			)

			if i % 100 == 0:
				model.eval()

				sample = img[:sample_size]

				with torch.no_grad():
					out, _ = model(sample, rand_index=None, lam=lam)

				utils.save_image(
					torch.cat([sample, out], 0),
					f"{args.sample_path}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
					nrow=sample_size,
					normalize=True,
					range=(-1, 1),
				)

				model.train()
	return avg_mse


def main(args):
	if dist.is_primary():
		sys.stdout = Logger(os.path.join(args.log_path, f'{args.dir_name}/log.txt'))
		print(args)
	device = "cuda"
	args.distributed = dist.get_world_size() > 1

	transform = transforms.Compose(
		[
			# transforms.RandomResizedCrop(args.size),
			transforms.RandomHorizontalFlip(0.5),
			transforms.Resize(args.size),
			transforms.CenterCrop(args.size),
			transforms.RandomGrayscale(p=0.2),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)

	dataset = datasets.ImageFolder(args.data_path, transform=transform)
	sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
	loader = DataLoader(
		dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0
	)

	model = VQVAE(embed_dim=args.embed_dim, n_embed=args.n_embed, noise=args.noise).to(device)

	if args.distributed:
		model = nn.parallel.DistributedDataParallel(
			model,
			device_ids=[dist.get_local_rank()],
			output_device=dist.get_local_rank(),
		)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	scheduler = None
	args.total_epoch = args.n_samples // (len(loader) * args.batch_size)
	if dist.is_primary():
		print('Number of Total Epochs: {}'.format(args.total_epoch))
	if args.sched == "cycle":
		scheduler = CycleScheduler(
			optimizer,
			args.lr,
			n_iter=len(loader) * args.total_epoch,
			momentum=None,
			warmup_proportion=0.05,
		)

	used_sample = 0
	epoch = 0
	best_mse= 9999
	best_epoch = 0
	while used_sample < args.n_samples:
		if args.distributed:
			sampler.set_epoch(epoch)
		avg_mse = train(epoch, loader, model, optimizer, scheduler, device, args)

		if dist.is_primary():
			if epoch % 5 == 0:
				torch.save(model.state_dict(), f"{args.ckpt_path}/vqvae_{str(epoch+1).zfill(3)}.pt")
			if avg_mse < best_mse:
				torch.save(model.state_dict(), f"{args.ckpt_path}/vqvae_best.pt")
				best_mse = avg_mse
				best_epoch = epoch
			print("Current MSE is {:.7f}, Best MSE is {:.7f} in Epoch {}".format(avg_mse, best_mse, best_epoch + 1))

		epoch += 1
		used_sample += len(loader) * args.batch_size


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--n_gpu", type=int, default=1)

	port = (
			2 ** 15
			+ 2 ** 14
			+ hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
	)
	parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

	parser.add_argument("--embed_dim", type=int, default=64)
	parser.add_argument("--n_embed", type=int, default=512)
	parser.add_argument("--noise", action='store_true')

	parser.add_argument("--size", type=int, default=256)
	parser.add_argument("--n_samples", type=int, default=10_000_000)
	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--sched", type=str)
	parser.add_argument("--batch_size", type=int, default=64, help="batch size for each gpu")

	parser.add_argument("--data_path", type=str)
	parser.add_argument("--log_path", type=str, default='logs')

	parser.add_argument("--input_noise", type=float, default=0.0)
	parser.add_argument("--mixup", action="store_true")
	parser.add_argument("--spherical", action="store_true")
	parser.add_argument("--ada", action="store_true")
	parser.add_argument("--ada_p", type=float, default=0.5)
	parser.add_argument("--cutmix", action="store_true")
	parser.add_argument('--beta', default=0, type=float,
						help='hyperparameter beta')
	parser.add_argument('--cutmix_prob', default=0, type=float,
						help='cutmix probability')

	parser.add_argument("--suffix", type=str)

	args = parser.parse_args()

	args.dir_name = "-".join([item for item in [
		args.suffix,
		f"embed{args.embed_dim}",
		f"nembed{args.n_embed}",
		"mixup" if args.mixup else None,
		"noise" if args.noise else None,
		"sp" if args.mixup and args.spherical else None,
		f"cutmix-{args.beta}-{args.cutmix_prob}" if args.cutmix else None,
		f"ada{args.ada_p}" if args.ada else None,
		f"noisein{args.input_noise}" if args.input_noise > 0 else None,
		] if item is not None])

	create_dir(os.path.join(args.log_path, args.dir_name))
	sample_path = os.path.join(args.log_path, f'{args.dir_name}/sample')
	ckpt_path = os.path.join(args.log_path, f'{args.dir_name}/checkpoint')
	create_dir(sample_path)
	create_dir(ckpt_path)
	args.sample_path = sample_path
	args.ckpt_path = ckpt_path

	dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
