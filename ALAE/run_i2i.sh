#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=3
python translation/adapt.py -c celeba-hq1024 \
	OUTPUT_DIR training_artifacts/ffhq \
	DATA.SOURCE male DATA.TARGET female \
	LANGEVIN.STEP 15 LANGEVIN.LR 1.0 \
	EBM.LR 0.01 EBM.LAYER 2 EBM.HIDDEN 1024 \
