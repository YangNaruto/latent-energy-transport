import numpy as np
import torch

def mixup_data(x, y=None, alpha=1.0, use_cuda=True, spherical=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    if spherical:
        mixed_x = lam * x + (1 - lam**2)**(0.5) * x[index, :]
    else:
        mixed_x = lam * x + (1 - lam) * x[index, :]
    # if y is not None:
    #     y_a, y_b = y, y[index]
    #     return mixed_x, y_a, y_b, lam
    # else:
    return mixed_x, index, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cut_mix(x, args):
    r = np.random.rand(1)
    lam = 1.0
    rand_index = torch.arange(0, x.size()[0]).cuda()
    if args.beta > 0 and r < args.cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(args.beta, args.beta)
        rand_index = torch.randperm(x.size()[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, rand_index, lam