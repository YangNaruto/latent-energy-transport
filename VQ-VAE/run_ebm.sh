#!/bin/bash
DATASET=$1
SRC=$2
TGT=$3
export CUDA_VISIBLE_DEVICES=2
python adapt.py --n_gpu 1 --dataset $DATASET --source $SRC --target $TGT \
	--channel_mul 8 --langevin_step 35 --langevin_lr 1.0 --lr 0.001  --beta1 0.5 --beta2 0.999 \
	--n_embed 512 --embed_dim 64 --batch_size 8 \
	--ae_ckpt ./logs/cat2dog/checkpoint/vqvae_best.pt \
	--data_root ./datasets/ \
