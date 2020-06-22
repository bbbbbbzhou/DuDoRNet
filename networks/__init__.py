import torch.nn as nn
import numpy as np
from utils import arange
from networks.networks import DRDN
import pdb


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)

    return network


def get_generator(name, opts):

    if name == 'DRDN':
        ic = 2
        if opts.use_prior:
            ic = ic + 2
        network = DRDN(n_channels=ic, G0=32, kSize=3, D=3, C=4, G=32, dilateSet=[1,2,3,3])

    else:
        raise NotImplementedError

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters: {}'.format(num_param))
    return set_gpu(network, opts.gpu_ids)