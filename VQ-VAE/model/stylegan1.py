import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.utils import spectral_norm
from math import sqrt
import math
from .swish import SwishModule


def init_linear(linear):
	init.xavier_normal(linear.weight)
	linear.bias.data.zero_()


def init_conv(conv, glu=True):
	init.kaiming_normal(conv.weight)
	if conv.bias is not None:
		conv.bias.data.zero_()


def get_activation(activation_fn):
	if activation_fn == 'lrelu':
		return nn.LeakyReLU(0.2)
	elif activation_fn == 'gelu':
		return nn.GELU()
	elif activation_fn == 'swish':
		return SwishModule()
	elif activation_fn == 'elu':
		return nn.ELU(alpha=1.0)
	elif activation_fn == 'celu':
		return nn.CELU(alpha=1.1)
	elif activation_fn == 'relu':
		return nn.ReLU()
	elif activation_fn == 'tanh':
		return nn.Tanh()


class EqualLR:
	def __init__(self, name):
		self.name = name

	def compute_weight(self, module):
		weight = getattr(module, self.name + '_orig')
		fan_in = weight.data.size(1) * weight.data[0][0].numel()

		return weight * sqrt(2 / fan_in)

	@staticmethod
	def apply(module, name):
		fn = EqualLR(name)

		weight = getattr(module, name)
		del module._parameters[name]
		module.register_parameter(name + '_orig', nn.Parameter(weight.data))
		module.register_forward_pre_hook(fn)

		return fn

	def __call__(self, module, input):
		weight = self.compute_weight(module)
		setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
	EqualLR.apply(module, name)

	return module


class FusedDownsample(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, padding=0):
		super().__init__()

		weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
		bias = torch.zeros(out_channel)

		fan_in = in_channel * kernel_size * kernel_size
		self.multiplier = sqrt(2 / fan_in)

		self.weight = nn.Parameter(weight, requires_grad=True)
		self.bias = nn.Parameter(bias, requires_grad=True)

		self.pad = padding

	def forward(self, x):
		weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
		weight = (
						 weight[:, :, 1:, 1:]
						 + weight[:, :, :-1, 1:]
						 + weight[:, :, 1:, :-1]
						 + weight[:, :, :-1, :-1]
				 ) / 4

		out = F.conv2d(x, weight, self.bias, stride=2, padding=self.pad)

		return out


def pixel_norm(x):
	return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
	@staticmethod
	def forward(ctx, grad_output, kernel, kernel_flip):
		ctx.save_for_backward(kernel, kernel_flip)

		grad_input = F.conv2d(
			grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
		)

		return grad_input

	@staticmethod
	def backward(ctx, gradgrad_output):
		kernel, kernel_flip = ctx.saved_tensors

		grad_input = F.conv2d(
			gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
		)

		return grad_input, None, None


class BlurFunction(Function):
	@staticmethod
	def forward(ctx, input, kernel, kernel_flip):
		ctx.save_for_backward(kernel, kernel_flip)

		output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

		return output

	@staticmethod
	def backward(ctx, grad_output):
		kernel, kernel_flip = ctx.saved_tensors

		grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

		return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
	def __init__(self, channel):
		super().__init__()

		weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
		weight = weight.view(1, 1, 3, 3)
		weight = weight / weight.sum()
		weight_flip = torch.flip(weight, [2, 3])

		self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
		self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

	def forward(self, input):
		return blur(input, self.weight, self.weight_flip)


# return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


# class EqualConv2d(nn.Module):
# 	def __init__(
# 			self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
# 	):
# 		super().__init__()
#
# 		self.weight = nn.Parameter(
# 			torch.randn(out_channel, in_channel, kernel_size, kernel_size)
# 		)
# 		self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
#
# 		self.stride = stride
# 		self.padding = padding
#
# 		if bias:
# 			self.bias = nn.Parameter(torch.zeros(out_channel))
#
# 		else:
# 			self.bias = None
#
# 	def forward(self, input):
# 		out = F.conv2d(
# 			input,
# 			self.weight * self.scale,
# 			bias=self.bias,
# 			stride=self.stride,
# 			padding=self.padding,
# 		)
#
# 		return out
#
# 	def __repr__(self):
# 		return (
# 			f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
# 			f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
# 		)
#


class EqualConv2d(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()

		conv = nn.Conv2d(*args, **kwargs)
		conv.weight.data.normal_()
		if conv.bias is not None:
			conv.bias.data.zero_()

		self.conv = equal_lr(conv)

	def forward(self, input):
		return self.conv(input)


class EqualLinear(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()

		linear = nn.Linear(in_dim, out_dim)
		linear.weight.data.normal_()
		linear.bias.data.zero_()

		self.linear = equal_lr(linear)

	def forward(self, input):
		return self.linear(input)


class ConvBlock(nn.Module):
	def __init__(
			self,
			in_channel,
			out_channel,
			kernel_size,
			padding,
			kernel_size2=None,
			padding2=None,
			downsample=False,
			fused=False,
			activation_fn='lrelu',
			last_layer=False
	):
		super().__init__()

		self.activation = get_activation(activation_fn=activation_fn)
		self.last = last_layer
		pad1 = padding
		pad2 = padding
		if padding2 is not None:
			pad2 = padding2

		kernel1 = kernel_size
		kernel2 = kernel_size
		if kernel_size2 is not None:
			kernel2 = kernel_size2

		self.conv1 = nn.Sequential(
			EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
		)

		if downsample:
			if fused:
				self.conv2 = nn.Sequential(
					# Blur(out_channel),
					FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
				)

			else:
				self.conv2 = nn.Sequential(
					# Blur(out_channel),
					EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
					nn.AvgPool2d(2),
				)
			if not self.last:
				self.skip = nn.Sequential(EqualConv2d(in_channel, out_channel, 1), nn.AvgPool2d(2))
		else:
			self.conv2 = nn.Sequential(
				EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
			)
			if not self.last:
				self.skip = nn.Sequential(EqualConv2d(in_channel, out_channel, 1))


	def forward(self, input, res=True):
		out = self.conv1(input)
		out = self.activation(out)

		out = self.conv2(out)
		out = self.activation(out)
		if not self.last:
			skip = self.skip(input)
			out = (skip + out) / math.sqrt(2)

		return out


class SNConvBlock(nn.Module):
	def __init__(
			self,
			in_channel,
			out_channel,
			kernel_size,
			padding,
			kernel_size2=None,
			padding2=None,
			downsample=False,
			fused=False,
			activation_fn='lrelu',
			n_classes=10,
			last_layer=False
	):
		super().__init__()

		self.activation = get_activation(activation_fn=activation_fn)
		self.last = last_layer
		pad1 = padding
		pad2 = padding
		if padding2 is not None:
			pad2 = padding2

		kernel1 = kernel_size
		kernel2 = kernel_size
		if kernel_size2 is not None:
			kernel2 = kernel_size2

		self.conv1 = nn.Sequential(
			spectral_norm(nn.Conv2d(in_channel, out_channel, kernel1, padding=pad1)),
		)

		if downsample:
			if fused:
				self.conv2 = nn.Sequential(
					# Blur(out_channel),
					FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
				)

			else:
				self.conv2 = nn.Sequential(
					# Blur(out_channel),
					spectral_norm(nn.Conv2d(out_channel, out_channel, kernel2, padding=pad2)),
					nn.AvgPool2d(2),
				)
			if not self.last:
				self.skip = nn.Sequential(EqualConv2d(in_channel, out_channel, 1), nn.AvgPool2d(2))
		else:
			self.conv2 = nn.Sequential(
				spectral_norm(nn.Conv2d(out_channel, out_channel, kernel2, padding=pad2)),
			)
			if not self.last:
				self.skip = nn.Sequential(EqualConv2d(in_channel, out_channel, 1))

	def forward(self, input: torch.Tensor, res=False, label=None):
		out = self.conv1(input)

		out = self.activation(out)

		out = self.conv2(out)
		out = self.activation(out)
		if not self.last:
			skip = self.skip(input)
			out = (skip + out) / math.sqrt(2)

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
	def __init__(self, size=64, channel_multiplier=1, input_channel=128, spectral=False,
				 add_attention=True, projection=False,
				 activation_fn='swish', cam=False):
		super().__init__()
		channels = {
			4: min(64 * channel_multiplier, 512),
			8: min(64 * channel_multiplier, 512),
			16: min(32 * channel_multiplier, 512),
			32: min(32 * channel_multiplier, 512),
			64: 32 * channel_multiplier,
			128: 32 * channel_multiplier,
			256: 16 * channel_multiplier,
			512: 8 * channel_multiplier,
			1024: 4 * channel_multiplier,
		}

		self.activation = get_activation(activation_fn=activation_fn)
		self.stddev_group = 4
		self.stddev_feat = 1
		convs = [EqualConv2d(input_channel, channels[size], 1)]
		log_size = int(math.log(size, 2))
		in_channel = channels[size]
		for i in range(log_size, 2, -1):
			out_channel = channels[2 ** (i - 1)]
			# fused = False if 2 ** i < 128 else True
			fused = False
			if spectral:
				convs.append(SNConvBlock(in_channel, out_channel, 3, 1, downsample=True, activation_fn=activation_fn,
										 fused=fused))
			else:
				convs.append(
					ConvBlock(in_channel, out_channel, 3, 1, downsample=True, activation_fn=activation_fn, fused=fused))
			if add_attention and i == 4 :
				convs.append(Attention(out_channel))
			in_channel = out_channel
		self.convs = nn.Sequential(*convs)
		if spectral:
			self.final_conv = SNConvBlock(in_channel, channels[4], 3, 1, 4, 0, activation_fn=activation_fn,
										  last_layer=True)
		else:
			self.final_conv = ConvBlock(in_channel, channels[4], 3, 1, 4, 0, activation_fn=activation_fn,
										last_layer=True)
		if projection:
			self.linear = nn.Sequential(EqualLinear(channels[4], channels[4] * 16), self.activation,
										EqualLinear(channels[4] * 16, 1))
		else:
			self.linear = EqualLinear(channels[4], 1)
		self.cam = cam
		if cam:
			if spectral:
				self.gap_fc = nn.utils.spectral_norm(nn.Linear(64 * channel_multiplier, 1, bias=False))
				self.gmp_fc = nn.utils.spectral_norm(nn.Linear(64 * channel_multiplier, 1, bias=False))
				self.final_conv_1 = EqualConv2d(64 * channel_multiplier, 64 * channel_multiplier, 1)
				self.conv1x1 = nn.Conv2d(128 * channel_multiplier, 64 * channel_multiplier, kernel_size=1, stride=1, bias=True)
				self.leaky_relu = nn.LeakyReLU(0.2, True)

				self.pad = nn.ReflectionPad2d(1)
				self.conv = nn.utils.spectral_norm(
					nn.Conv2d(64 * channel_multiplier, 1, kernel_size=4, stride=1, padding=0, bias=False))
			else:
				self.gap_fc = nn.Linear(64 * channel_multiplier, 1, bias=False)
				self.gmp_fc = nn.Linear(64 * channel_multiplier, 1, bias=False)
				self.final_conv_1 = EqualConv2d(64 * channel_multiplier, 64 * channel_multiplier, 1)
				self.conv1x1 = nn.Conv2d(128 * channel_multiplier, 64 * channel_multiplier, kernel_size=1, stride=1, bias=True)
				self.leaky_relu = nn.LeakyReLU(0.2, True)

				self.pad = nn.ReflectionPad2d(1)
				self.conv = nn.Conv2d(64 * channel_multiplier, 1, kernel_size=4, stride=1, padding=0, bias=False)

	def forward(self, input, label=None):
		out = self.convs(input)
		batch, channel, height, width = out.shape
		# group = min(batch, self.stddev_group)
		# stddev = out.view(
		#  group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
		# )
		# stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
		# stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
		# stddev = stddev.repeat(group, 1, height, width)
		# out = torch.cat([out, stddev], 1)
		# h = self.final_conv(out)
		# h = torch.flatten(h, start_dim=1)
		# out = self.linear(h)
		# if label is not None:
		# 	out = out + torch.sum(self.embedding(label) * h, 1, keepdim=True)
		#


		assert out is not None
		if not self.cam:
			h = self.final_conv(out, res=False)
			h = torch.flatten(h, start_dim=1)

			out = self.linear(h)
			if label is not None:
				out = out + torch.sum(self.embedding(label) * h, 1, keepdim=True)
		else:
			# print('hee')
			gap = torch.nn.functional.adaptive_avg_pool2d(out, 1)
			gap_logit = self.gap_fc(gap.view(out.shape[0], -1))
			gap_weight = list(self.gap_fc.parameters())[0]
			gap = out * gap_weight.unsqueeze(2).unsqueeze(3)

			gmp = torch.nn.functional.adaptive_max_pool2d(out, 1)
			gmp_logit = self.gmp_fc(gmp.view(out.shape[0], -1))
			gmp_weight = list(self.gmp_fc.parameters())[0]
			gmp = out * gmp_weight.unsqueeze(2).unsqueeze(3)

			cam_logit = torch.cat([gap_logit, gmp_logit], 1)
			x = torch.cat([gap, gmp], 1)
			x = self.leaky_relu(self.conv1x1(x))

			heatmap = torch.sum(x, dim=1, keepdim=True)

			x = self.pad(x)
			out = self.conv(x)
		return out
