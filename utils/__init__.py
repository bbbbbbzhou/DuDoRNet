import os
import os.path as path
import cv2
import numpy as np
import yaml
from scipy.fftpack import fft
from PIL import Image
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize

from .misc import *
from .argparser import *


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))


def tensor_to_np(img_var):
    """

    From 1 x C x W x H [0..1] to C x W x H [0..1]
    """
    img_np = img_var[0].cpu().numpy()
    return img_np


def np_to_pil(img_np):
    """

    From C x W x H [0..1] to W x H x C [0..255]
    """
    img = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        img = img[0]
    else:
        img = img.transpose((1, 2, 0))

    return Image.fromarray(img)


def display_transform(img_var):
    f = Compose([
        tensor_to_np,
        np_to_pil,
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])
    return f(img_var)


def next_power_of_2(n):
    if n == 0:
        return 1
    if n & (n - 1) == 0:
        return n
    while n & (n - 1) > 0:
        n &= (n - 1)
    return n << 1


def design_filter(detector_length, filt_type='ram-lak', d=1.0):
    order = max(64, next_power_of_2(2 * detector_length))
    n = np.arange(order/2 + 1)
    filtImpResp = np.zeros(order//2 + 1)
    filtImpResp[0] = 0.25
    filtImpResp[1::2] = -1 / ((np.pi * n[1::2]) ** 2)
    filtImpResp = np.concatenate(
        [filtImpResp, filtImpResp[len(filtImpResp) - 2:0:-1]]
    )
    filt = 2 * fft(filtImpResp).real
    filt = filt[:order//2 + 1]
    w = 2 * np.pi * np.arange(filt.shape[0]) / order

    if filt_type == 'ram-lak':
        pass
    elif filt_type == 'shepp-logan':
        filt[1:] *= np.sin(w[1:] / (2 * d)) / (w[1:] / (2 * d))
    elif filt_type == 'cosine':
        filt[1:] *= np.cos(w[1:] / (2 * d))
    elif filt_type == 'hamming':
        filt[1:] *= (0.54 + 0.46 * np.cos(w[1:] / d))
    elif filt_type == 'hann':
        filt[1:] *= (1 + np.cos(w[1:] / d)) / 2
    else:
        raise ValueError("Invalid filter type")

    filt[w > np.pi * d] = 0.0
    filt = np.concatenate([filt, filt[len(filt) - 2:0:-1]])
    return filt


def arange(start, stop, step):
    """ Matlab-like arange
    """
    r = list(np.arange(start, stop, step).tolist())
    if r[-1] + step == stop:
        r.append(stop)
    return np.array(r)
