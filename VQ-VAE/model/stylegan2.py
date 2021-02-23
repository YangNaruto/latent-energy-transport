import math
import random
import functools
import operator
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.utils import spectral_norm


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
	rest_dim = [1] * (input.ndim - bias.ndim - 1)
	return (
			F.leaky_relu(
				input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
			)
			* scale
	)


class FusedLeakyReLU(nn.Module):
	def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
		super().__init__()

		self.bias = nn.Parameter(torch.zeros(channel))
		self.negative_slope = negative_slope
		self.scale = scale

	def forward(self, input):
		return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


class ScaledLeakyReLU(nn.Module):
	def __init__(self, negative_slope=0.2):
		super().__init__()

		self.negative_slope = negative_slope

	def forward(self, input):
		out = F.leaky_relu(input, negative_slope=self.negative_slope)

		return out * math.sqrt(2)


class Blur(nn.Module):
	def __init__(self, channels):
		super(Blur, self).__init__()
		f = np.array([1, 2, 1], dtype=np.float32)
		f = f[:, np.newaxis] * f[np.newaxis, :]
		f /= np.sum(f)
		kernel = torch.Tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
		self.register_buffer('weight', kernel)
		self.groups = channels

	def forward(self, x):
		return F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)


class EqualConv2d(nn.Module):
	def __init__(
			self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
	):
		super().__init__()

		self.weight = nn.Parameter(
			torch.randn(out_channel, in_channel, kernel_size, kernel_size)
		)
		self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

		self.stride = stride
		self.padding = padding

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_channel))

		else:
			self.bias = None

	def forward(self, input):
		out = F.conv2d(
			input,
			self.weight * self.scale,
			bias=self.bias,
			stride=self.stride,
			padding=self.padding,
		)

		return out

	def __repr__(self):
		return (
			f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
			f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
		)


class EqualLinear(nn.Module):
	def __init__(
			self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
	):
		super().__init__()

		self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

		else:
			self.bias = None

		self.activation = activation

		self.scale = (1 / math.sqrt(in_dim)) * lr_mul
		self.lr_mul = lr_mul

	def forward(self, input):
		if self.activation:
			out = F.linear(input, self.weight * self.scale)
			out = fused_leaky_relu(out, self.bias * self.lr_mul)

		else:
			out = F.linear(
				input, self.weight * self.scale, bias=self.bias * self.lr_mul
			)

		return out

	def __repr__(self):
		return (
			f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
		)


class NoiseInjection(nn.Module):
	def __init__(self):
		super().__init__()

		self.weight = nn.Parameter(torch.zeros(1))

	def forward(self, image, noise=None):
		if noise is None:
			batch, _, height, width = image.shape
			noise = image.new_empty(batch, 1, height, width).normal_()

		return image + self.weight * noise


class ConvLayer(nn.Module):
	def __init__(
			self,
			in_channel,
			out_channel,
			kernel_size,
			downsample=False,
			blur_kernel=[1, 3, 3, 1],
			bias=True,
			activate=True,
			sn=False,
			blur=False
	):
		super().__init__()
		layers = nn.ModuleList()

		if downsample:
			factor = 2
			p = (len(blur_kernel) - factor) + (kernel_size - 1)
			pad0 = (p + 1) // 2
			pad1 = p // 2
			if blur:
				layers.append(Blur(in_channel))

			stride = 2
			# self.padding = 0
			self.padding = kernel_size // 2

		else:
			stride = 1
			self.padding = kernel_size // 2
		if sn:
			layers.append(
				spectral_norm(nn.Conv2d(
					in_channel,
					out_channel,
					kernel_size,
					padding=self.padding,
					stride=stride,
					bias=bias and not activate,
				))
			)
		else:
			layers.append(
				EqualConv2d(
					in_channel,
					out_channel,
					kernel_size,
					padding=self.padding,
					stride=stride,
					bias=bias and not activate,
				)
			)

		self.activate = None
		if activate:
			# layers.append(nn.LeakyReLU(0.2))
			if bias:
				self.activate = FusedLeakyReLU(out_channel)

			else:
				self.activate = ScaledLeakyReLU(0.2)

		self.conv = nn.Sequential(*layers)
		self.noise = NoiseInjection()

	def forward(self, x, noise=None):
		x = self.conv(x)
		# x = self.noise(x, noise=noise)
		if self.activate is not None:
			x = self.activate(x)
		return x


class ResBlock(nn.Module):
	def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], sn=False, blur=True):
		super().__init__()

		self.conv1 = ConvLayer(in_channel, in_channel, 3, sn=sn, blur=blur)
		self.conv2 = ConvLayer(in_channel, out_channel, 3, sn=sn, downsample=True, blur=blur)

		self.skip = ConvLayer(
			in_channel, out_channel, 1, downsample=True, activate=False, bias=False, blur=blur
		)

	def forward(self, input):
		out = self.conv1(input)
		out = self.conv2(out)

		skip = self.skip(input)
		out = (out + skip) / math.sqrt(2)

		return out

class Attention(nn.Module):
	""" Self attention Layer"""

	def __init__(self, in_dim, activation=None):
		super(Attention, self).__init__()
		self.chanel_in = in_dim
		self.activation = activation

		self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax = nn.Softmax(dim=-1)  #

	def forward(self, x):
		"""
			inputs :
				x : input feature maps( B X C X W X H)
			returns :
				out : self attention value + input feature
				attention: B X N X N (N is Width*Height)
		"""
		m_batchsize, C, width, height = x.size()
		proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
		proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
		energy = torch.bmm(proj_query, proj_key)  # transpose check
		attention = self.softmax(energy)  # BX (N) X (N)
		proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

		out = torch.bmm(proj_value, attention.permute(0, 2, 1))
		out = out.view(m_batchsize, C, width, height)

		out = self.gamma * out + x
		return out

class EBM(nn.Module):
	def __init__(self, size=64, channel_multiplier=4, input_channel=128, spectral=False,
				 add_attention=False, cam=False, blur=True, blur_kernel=[1, 3, 3, 1]):
		super().__init__()
		channels = {
			4: min(64 * channel_multiplier, 512),
			8: min(64 * channel_multiplier, 512),
			16: min(64 * channel_multiplier, 512),
			32: min(64 * channel_multiplier, 512),
			64: 32 * channel_multiplier,
			128: 16 * channel_multiplier,
			256: 8 * channel_multiplier,
			512: 4 * channel_multiplier,
			1024: 2 * channel_multiplier,
		}

		convs = [ConvLayer(input_channel, channels[size], 1, sn=spectral, blur=blur)]
		log_size = int(math.log(size, 2))
		in_channel = channels[size]
		for i in range(log_size, 2, -1):
			out_channel = channels[2 ** (i - 1)]
			convs.append(ResBlock(in_channel, out_channel, blur_kernel=blur_kernel, sn=spectral, blur=blur))
			if add_attention and i == 4 :
				convs.append(Attention(out_channel))
			in_channel = out_channel
		self.convs = nn.Sequential(*convs)
		self.stddev_group = 4
		self.stddev_feat = 1
		self.final_conv = ConvLayer(in_channel, channels[4], 3, sn=spectral, blur=blur)
		self.final_linear = nn.Sequential(
			EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
			EqualLinear(channels[4], 1),
		)

	def make_noise(self):
		device = self.input.input.device
		noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
		for i in range(3, self.log_size + 1):
			for _ in range(2):
				noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
		return noises

	def forward(self, input):
		out = self.convs(input)
		batch, channel, height, width = out.shape
		# group = min(batch, self.stddev_group)
		# stddev = out.view(
		# 	group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
		# )
		# stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
		# stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
		# stddev = stddev.repeat(group, 1, height, width)
		# out = torch.cat([out, stddev], 1)
		out = self.final_conv(out)
		out = out.view(batch, -1)
		out = self.final_linear(out)
		return out
