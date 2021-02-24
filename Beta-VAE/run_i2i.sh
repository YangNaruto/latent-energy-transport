#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python translation.py --ckpt checkpoints/celeba_H_beta10_z32/celeba/best --z_dim 32 --source male --target female \
	--langevin_step 10 --lr 0.5 \
