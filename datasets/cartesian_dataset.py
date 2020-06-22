import os
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from datasets.utilizes import *
from models.utils import fft2, ifft2, to_tensor


class MRIDataset_Cartesian(data.Dataset):
    def __init__(self, opts, mode):
        """
        Args:
            data_dir: data folder for retrieving
                1) Ref: T1 kspace data
                2) Tag: T2 / FLAIR kspace data
        """
        self.mode = mode
        if self.mode == 'TRAIN':
            self.data_dir_ref = os.path.join(opts.data_root, 'TRAIN', opts.protocol_ref)
            self.data_dir_tag = os.path.join(opts.data_root, 'TRAIN', opts.protocol_tag)

            self.center_fractions = np.concatenate((1.0 / 10.0 * np.ones(1000), 1.0 / np.linspace(1, 8, 100)), axis=0)          # central fraction fix to 1/8
            self.accelerations = np.concatenate((np.random.randint(1000, 8000, 1000) / 1000, np.linspace(1, 8, 100)), axis=0)   # acc range from (8.0 ~ 1.0)
            self.seed = None

        if self.mode == 'VALI':
            self.data_dir_ref = os.path.join(opts.data_root, 'VALI', opts.protocol_ref)
            self.data_dir_tag = os.path.join(opts.data_root, 'VALI', opts.protocol_tag)

            self.center_fractions = [opts.center_fractions]  # central fraction fix to 1/8
            self.accelerations = [opts.accelerations]  # fix acc
            self.seed = 1234

        if self.mode == 'TEST':
            self.data_dir_ref = os.path.join(opts.data_root, 'TEST', opts.protocol_ref)
            self.data_dir_tag = os.path.join(opts.data_root, 'TEST', opts.protocol_tag)

            self.center_fractions = [opts.center_fractions]  # central fraction fix to 1/8
            self.accelerations = [opts.accelerations]  # fix acc
            self.seed = 5678

        self.protocol_ref = opts.protocol_ref
        self.protocol_tag = opts.protocol_tag

        self.mask_func = MaskFunc_Cartesian(self.center_fractions, self.accelerations)
        self.data_dir_ref_kspace = os.path.join(self.data_dir_ref, 'kspace')    # ref kspace directory (T1)
        self.data_dir_tag_kspace = os.path.join(self.data_dir_tag, 'kspace')    # tag kspace directory (T2 / FLAIR)

        self.filenames_ref_kspace = sorted(os.listdir(self.data_dir_ref_kspace))
        self.filenames_tag_kspace = sorted(os.listdir(self.data_dir_tag_kspace))

    def __getitem__(self, idx):
        """
        Args:
            index: the index of MRI slice kspace

        Returns:
            1) Ref: T1_full kspace & image
            2) Ref: T1_subsampled kspace & image
            3) Tag: T2 / FLAIR _full kspace & image
            4) Tag: T2 / FLAIR _subsampled kspace & image
        """

        ## Ref: T1
        ref_kspace_raw = sio.loadmat(os.path.join(self.data_dir_ref_kspace, self.filenames_ref_kspace[idx]))['kspace_py']
        ref_kspace_full = to_tensor(ref_kspace_raw).float()
        ref_image_full = ifft2(ref_kspace_full).permute(2, 0, 1)  # Full Recon Ref - T1

        ref_kspace_sub, ref_kspace_mask2d = apply_mask(data=ref_kspace_full.float(), mask_func=self.mask_func, seed=self.seed)
        ref_image_sub = ifft2(ref_kspace_sub).permute(2, 0, 1)  # Sub Recon Ref - T1

        ## Tag: T2 / FLAIR
        tag_kspace_raw = sio.loadmat(os.path.join(self.data_dir_tag_kspace, self.filenames_tag_kspace[idx]))['kspace_py']
        tag_kspace_full = to_tensor(tag_kspace_raw).float()
        tag_image_full = ifft2(tag_kspace_full).permute(2, 0, 1)  # Full Recon T2 / FLAIR

        tag_kspace_sub, tag_kspace_mask2d = apply_mask(data=tag_kspace_full.float(), mask_func=self.mask_func, seed=self.seed)
        tag_image_sub = ifft2(tag_kspace_sub).permute(2, 0, 1)  # Sub Recon T2 / FLAIR

        return {'ref_kspace_full': ref_kspace_full, 'ref_kspace_sub': ref_kspace_sub, 'ref_image_full': ref_image_full, 'ref_image_sub': ref_image_sub,
                'ref_kspace_mask2d': ref_kspace_mask2d,
                'tag_kspace_full': tag_kspace_full, 'tag_kspace_sub': tag_kspace_sub, 'tag_image_full': tag_image_full, 'tag_image_sub': tag_image_sub,
                'tag_kspace_mask2d': tag_kspace_mask2d,
                'accelerations': self.accelerations}

    def __len__(self):
        return len(self.filenames_ref_kspace)


if __name__ == '__main__':
    a = 1
