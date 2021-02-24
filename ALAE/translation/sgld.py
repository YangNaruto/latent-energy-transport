import torch
from torch.optim import Optimizer
import numpy as np


class SGLD(Optimizer):
	""" Stochastic Gradient Langevin Dynamics Sampler with preconditioning.
		Optimization variable is viewed as a posterior sample under Stochastic
		Gradient Langevin Dynamics with noise rescaled in eaach dimension
		according to RMSProp.
	"""
	def __init__(self,
				 params,
				 lr=1e-2,
				std_dev=0.0, decay=None) -> None:
		""" Set up a SGLD Optimizer.

		Parameters
		----------
		params : iterable
			Parameters serving as optimization variable.
		lr : float, optional
			Base learning rate for this optimizer.
			Must be tuned to the specific function being minimized.
			Default: `1e-2`.


		"""
		if lr < 0.0:
			raise ValueError("Invalid learning rate: {}".format(lr))

		defaults = dict(
			lr=lr, std_dev=std_dev
		)
		super().__init__(params, defaults)


	def step(self, closure=None):
		loss = None

		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for parameter in group["params"]:

				if parameter.grad is None:
					continue

				state = self.state[parameter]
				lr = group["lr"]
				std_dev = group["std_dev"]
				gradient = parameter.grad.data

				#  State initialization {{{ #

				if len(state) == 0:
					state["iteration"] = 0
					state["momentum"] = torch.ones_like(parameter)

				#  }}} State initialization #
				current_std_dev = self.polynomial(state["iteration"], 15, std_dev)
				noise = current_std_dev * torch.ones_like(parameter.grad)
				state["iteration"] += 1

				parameter.data.add_(-lr * gradient + noise)

		return loss

	def cyclic(self, T, i, lr, M=4, min_lr=0.):
		rcounter = T + i
		cos_inner = np.pi * (rcounter % (T // M))
		cos_inner /= T // M
		cos_out = np.cos(cos_inner) + 1
		lr = float(np.clip(0.5 * cos_out * lr, min_lr, 100))
		return lr

	def polynomial(self, t, T, base_lr, end_lr=0.0001, power=1.):
		lr = (base_lr - end_lr) * ((1 - t / T) ** power) + end_lr

		# lr = a * (b + t) ** power
		return lr
