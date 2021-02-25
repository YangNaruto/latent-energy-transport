# Implemention of EBM for I2I based on ALAE

This is the implementation based on ALAE which is adapted from the official repo of [ALAE](https://github.com/podgorskiy/ALAE). Only the essential instruction is given below.

<p align="left"><img width="95%" src="assets/celeba-hq.png" /></p>
<p align="left"><img width="95%" src="assets/afhq.png" /></p>

## Repository organization

#### Running scripts

The code in the repository is organized in such a way that all scripts must be run from the root of the repository.
If you use an IDE (e.g. PyCharm or Visual Studio Code), just set *Working Directory* to point to the root of the repository.

If you want to run from the command line, then you also need to set **PYTHONPATH** variable to point to the root of the repository.

For example, let's say we've cloned repository to *~/ALAE* directory, then do:

    $ cd ~/ALAE
    $ export PYTHONPATH=$PYTHONPATH:$(pwd)

![pythonpath](https://podgorskiy.com/static/pythonpath.svg)

<!-- #### Repository structure

| Path | Description
| :--- | :----------
| ALAE | Repository root folder
| &boxvr;&nbsp; configs | Folder with yaml config files.
| &boxv;&nbsp; &boxvr;&nbsp; bedroom.yaml | Config file for LSUN bedroom dataset at 256x256 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; celeba.yaml | Config file for CelebA dataset at 128x128 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; celeba-hq256.yaml | Config file for CelebA-HQ dataset at 256x256 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; celeba_ablation_nostyle.yaml | Config file for CelebA 128x128 dataset for ablation study (no styles).
| &boxv;&nbsp; &boxvr;&nbsp; celeba_ablation_separate.yaml | Config file for CelebA 128x128 dataset for ablation study (separate encoder and discriminator).
| &boxv;&nbsp; &boxvr;&nbsp; celeba_ablation_z_reg.yaml | Config file for CelebA 128x128 dataset for ablation study (regress in Z space, not W).
| &boxv;&nbsp; &boxvr;&nbsp; ffhq.yaml | Config file for FFHQ dataset at 1024x1024 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; mnist.yaml | Config file for MNIST dataset using Style architecture.
| &boxv;&nbsp; &boxur;&nbsp; mnist_fc.yaml | Config file for MNIST dataset using only fully connected layers (Permutation Invariant MNIST).
| &boxvr;&nbsp; dataset_preparation | Folder with scripts for dataset preparation.
| &boxv;&nbsp; &boxvr;&nbsp; prepare_celeba_hq_tfrec.py | To prepare TFRecords for CelebA-HQ dataset at 256x256 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; prepare_celeba_tfrec.py | To prepare TFRecords for CelebA dataset at 128x128 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; prepare_mnist_tfrec.py | To prepare TFRecords for MNIST dataset.
| &boxv;&nbsp; &boxvr;&nbsp; split_tfrecords_bedroom.py | To split official TFRecords from StyleGAN paper for LSUN bedroom dataset.
| &boxv;&nbsp; &boxur;&nbsp; split_tfrecords_ffhq.py | To split official TFRecords from StyleGAN paper for FFHQ dataset.
| &boxvr;&nbsp; dataset_samples | Folder with sample inputs for different datasets. Used for figures and for test inputs during training.
| &boxvr;&nbsp; make_figures | Scripts for making various figures.
| &boxvr;&nbsp; metrics | Scripts for computing metrics.
| &boxvr;&nbsp; principal_directions | Scripts for computing principal direction vectors for various attributes. **For interactive demo**.
| &boxvr;&nbsp; style_mixing | Sample inputs and script for producing style-mixing figures.
| &boxvr;&nbsp; training_artifacts | Default place for saving checkpoints/sample outputs/plots.
| &boxv;&nbsp; &boxur;&nbsp; download_all.py | Script for downloading all pretrained models.
| &boxvr;&nbsp; interactive_demo.py | Runnable script for interactive demo.
| &boxvr;&nbsp; train_alae.py | Runnable script for training.
| &boxvr;&nbsp; train_alae_separate.py | Runnable script for training for ablation study (separate encoder and discriminator).
| &boxvr;&nbsp; checkpointer.py | Module for saving/restoring model weights, optimizer state and loss history.
| &boxvr;&nbsp; custom_adam.py | Customized adam optimizer for learning rate equalization and zero second beta.
| &boxvr;&nbsp; dataloader.py | Module with dataset classes, loaders, iterators, etc.
| &boxvr;&nbsp; defaults.py | Definition for config variables with default values.
| &boxvr;&nbsp; launcher.py | Helper for running multi-GPU, multiprocess training. Sets up config and logging.
| &boxvr;&nbsp; lod_driver.py | Helper class for managing growing/stabilizing network.
| &boxvr;&nbsp; lreq.py | Custom `Linear`, `Conv2d` and `ConvTranspose2d` modules for learning rate equalization.
| &boxvr;&nbsp; model.py | Module with high-level model definition.
| &boxvr;&nbsp; model_separate.py | Same as above, but for ablation study.
| &boxvr;&nbsp; net.py | Definition of all network blocks for multiple architectures.
| &boxvr;&nbsp; registry.py | Registry of network blocks for selecting from config file.
| &boxvr;&nbsp; scheduler.py | Custom schedulers with warm start and aggregating several optimizers.
| &boxvr;&nbsp; tracker.py | Module for plotting losses.
| &boxur;&nbsp; utils.py | Decorator for async call, decorator for caching, registry for network blocks. -->

#### Datasets

~~Training is done using TFRecords. TFRecords are read using [DareBlopy](https://github.com/podgorskiy/DareBlopy), which allows using them with Pytorch.~~ We fail to deploy it. So, we prepare the LMDB dataset

    python prepare_data.py --out ./data/datasets/DATASET_NAME --n_worker N_WORKER RAW_DATASET_PATH
Above command should generate two files under the folder `./data/datasets/DATASET_NAME`: `data.lmdb` and `lock.lmdb`.

In config files as well as in all preparation scripts, it is assumed that all datasets are in `./data/datasets/`. You can either change path in config files, either create a symlink to where you store datasets.

**Note that, the raw image files should be also put under `./datasets/DATASET_NAME`.**

The official way of generating CelebA-HQ can be challenging. Please refer to this page: https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download
You can get the pre-generated dataset from: https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P

#### Pre-trained models

To download pre-trained models from original ALAE run:

    python training_artifacts/download_all.py

**Note**: There used to be problems with downloading models from Google Drive due to download limit. 
Now, the script is setup in a such way that if it fails to download data from Google Drive it will try to download it from S3.

If you experience problems, try deleting all *.pth files, updating *dlutils* package (`pip install dlutils --upgrade`) and then run `download_all.py` again.
If that does not solve the problem, please open an issue. Also, you can try downloading models manually from here: https://drive.google.com/drive/folders/1tsI1q1u8QRX5t7_lWCSjpniLGlNY-3VY?usp=sharing


In config files, `OUTPUT_DIR` points to where weights are saved to and read from. For example: `OUTPUT_DIR: training_artifacts/celeba-hq256`

In `OUTPUT_DIR` it saves a file `last_checkpoint` which contains path to the actual `.pth` pickle with model weight. If you want to test the model with a specific weight file, you can simply modify `last_checkpoint` file.

## Stage 1: Train ALAE

In addition to installing required packages:

    pip install -r requirements.txt

~~You will need to install [DareBlopy](https://github.com/podgorskiy/DareBlopy):~~

    pip install dareblopy

To run training:

    python train_alae.py -c <config>
or

    sh run_alae.sh
    
It will run multi-GPU training on all available GPUs. It uses `DistributedDataParallel` for parallelism. 
If only one GPU available, it will run on single GPU, no special care is needed.

The recommended number of GPUs is 8. Reproducibility on a smaller number of GPUs may have issues. You might need to adjust the batch size in the config file depending on the memory size of the GPUs.

In config files, `OUTPUT_DIR` points to where weights are saved to and read from. For example: `OUTPUT_DIR: training_artifacts/celeba-hq256`

In `OUTPUT_DIR` it saves a file `last_checkpoint` which contains path to the actual `.pth` pickle with model weight. If you want to test the model with a specific weight file, you can simply modify `last_checkpoint` file.

## Stage 2: Train EBM

    sh run_i2i.sh

* `-c`: configuration file loaded from directory `configs`.
* `OUTPUT_DIR`: the directory containing the pretrained checkpoint
* `DATA.SOURCE`: translation source
* `DATA.TARGET`: translation target
* `LANGEVIN.STEP`: Langevin steps
* `LANGEVIN.LR`: Langevin step size
* `EBM.LR`: EBM learning rate
* `EBM.LAYER`: EBM hidden layers
* `EBM.HIDDEN`: EBM hidden dimension

The usage of configuration files is:

    AFHQ: afhq.yaml
    CelebA-HQ 1024x1024: celeba-hq1024.yaml
    CelebA-HQ 256x256: celeba-hq.yaml
Results will be stored in the directory `./results/{DATASET_NAME}/{DATA.SOURCE}2{DATA.TARGET}`.


## Citation

    @InProceedings{pidhorskyi2020adversarial,
     author   = {Pidhorskyi, Stanislav and Adjeroh, Donald A and Doretto, Gianfranco},
     booktitle = {Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)},
     title    = {Adversarial Latent Autoencoders},
     year     = {2020},
    }

    @article{zhao2020unpaired,
    title={Unpaired Image-to-Image Translation via Latent Energy Transport},
    author={Zhao, Yang and Chen, Changyou},
    journal={arXiv preprint arXiv:2012.00649},
    year={2020}
    }