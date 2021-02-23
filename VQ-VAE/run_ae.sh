#!/bin/bash
DATASET=$1
export CUDA_VISIBLE_DEVICES=1
python train_vqvae.py --data_path ./datasets/$DATASET/train \
	--n_gpu 1 --batch_size 64 --suffix $DATASET \
	--embed_dim 64 --n_embed 512 --input_noise 0.03 \
