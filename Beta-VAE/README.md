# Implemention of EBM for I2I based on BetaVAE

## Dependencies

```
visdom
```
<br>

## Usage

### Stage 1: Pretrain BetaVAE
```
sh run_celeba_H_beta10_z32.sh
```

### Stage 2: Train EBM
```
sh run_i2i.sh
```

## Acknowledgements

We adapt the the PyTorch reimplementation of BetaVAE from the [repo](https://github.com/1Konny/Beta-VAE).

## Citations
```
@article{zhao2020unpaired,
  title={Unpaired Image-to-Image Translation via Latent Energy Transport},
  author={Zhao, Yang and Chen, Changyou},
  journal={arXiv preprint arXiv:2012.00649},
  year={2020}
}

@inproceedings{Higgins2017betaVAELB,
  title={beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework},
  author={I. Higgins and Lo{\"i}c Matthey and A. Pal and C. Burgess and Xavier Glorot and M. Botvinick and S. Mohamed and Alexander Lerchner},
  booktitle={ICLR},
  year={2017}
}
```
