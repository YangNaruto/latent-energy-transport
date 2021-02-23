#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python extract_code.py --ckpt /home/yzhao/ideas/vq-vae-v0/logs/checkpoint/vqvae_1750.pt \
	--name selfie2anime \
	--data_path /home/yzhao/ProEBM/dataset/selfie2anime