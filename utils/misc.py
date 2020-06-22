__all__ = ['read_dir', 'get_last_checkpoint', 'compute_metrics', 'get_aapm_minmax',
    'convert_coefficient2hu', 'convert_hu2coefficient']

import os
import os.path as path
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from skimage.measure import compare_ssim, compare_psnr


def read_dir(dir_path, predicate=None, name_only=False):
    if predicate in {'dir', 'file'}:
        predicate = {
            'dir': lambda x: path.isdir(path.join(dir_path, x)),
            'file':lambda x: path.isfile(path.join(dir_path, x))
        }[predicate]

    return [f if name_only else path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if (True if predicate is None else predicate(f))]


def get_last_checkpoint(checkpoint_dir, predicate=None, pattern=None):
    if predicate is None:
        predicate = lambda x: x.endswith('pth') or x.endswith('pt')
    
    checkpoints = read_dir(checkpoint_dir, predicate)
    if len(checkpoints) == 0:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: path.getmtime(x))
    
    checkpoint = checkpoints[-1]
    if pattern is None:
        pattern = lambda x: int(path.basename(x).split('_')[-1].split('.')[0])
    return checkpoint, pattern(checkpoint)


def compute_metrics(lq_image, hq_image, metrics=None):
    psnr = compare_psnr(lq_image, hq_image, hq_image.max())
    ssim = compare_ssim(lq_image, hq_image, data_range=hq_image.max())

    if metrics is None:
        return {'psnr': [psnr], 'ssim': [ssim]}
    else:
        metrics['psnr'].append(psnr)
        metrics['ssim'].append(ssim)
        return metrics


def convert_coefficient2hu(image):
    image = (image - 0.0192) / 0.0192 * 1000
    return image


def convert_hu2coefficient(image):
    image = image * 0.0192 / 1000 + 0.0192
    return image


def get_aapm_minmax(data_dir,
    splits=('test', 'train', 'val'), tags=('dense_view', 'sparse_view')):
    data_files = []
    for s in splits:
        split_dir = path.join(data_dir, s)
        for d in os.listdir(split_dir):
            study_dir = path.join(split_dir, d)
            for f in os.listdir(study_dir):
                data_file = path.join(study_dir, f)
                if f.endswith('.mat'):
                    data_files.append(data_file)

    val_max = -float('inf')
    val_min = float('inf')
    for f in tqdm(data_files):
        data = sio.loadmat(f)
        data = np.array([data[t] for t in tags])

        if data.max() > val_max:
            val_max = data.max()
        if data.min() < val_min:
            val_min = data.min()

    return val_min, val_max