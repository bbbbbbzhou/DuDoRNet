import os
from math import log10
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from skimage.measure import compare_ssim as ssim

from networks import get_generator
from networks.networks import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, get_nonlinearity, DataConsistencyInKspace_I, DataConsistencyInKspace_K, fft2, complex_abs_eval

import pdb


class RecurrentModel(nn.Module):
    def __init__(self, opts):
        super(RecurrentModel, self).__init__()

        self.loss_names = []
        self.networks = []
        self.optimizers = []

        self.n_recurrent = opts.n_recurrent

        # set default loss flags
        loss_flags = ("w_img_L1")
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G_I = get_generator(opts.net_G, opts)
        self.net_G_K = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G_I)
        self.networks.append(self.net_G_K)

        if self.is_train:
            self.loss_names += ['loss_G_L1']
            param = list(self.net_G_I.parameters()) + list(self.net_G_K.parameters())
            self.optimizer_G = torch.optim.Adam(param,
                                                lr=opts.lr,
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)
            self.optimizers.append(self.optimizer_G)

        self.criterion = nn.L1Loss()

        self.opts = opts

        # data consistency layers in image space & k-space
        dcs_I = []
        for i in range(self.n_recurrent):
            dcs_I.append(DataConsistencyInKspace_I(noise_lvl=None))
        self.dcs_I = dcs_I

        dcs_K = []
        for i in range(self.n_recurrent):
            dcs_K.append(DataConsistencyInKspace_K(noise_lvl=None))
        self.dcs_K = dcs_K

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        self.ref_kspace_full = data['ref_kspace_full'].to(self.device)
        self.ref_kspace_sub = data['ref_kspace_sub'].to(self.device)
        self.ref_image_full = data['ref_image_full'].to(self.device)
        self.ref_image_sub = data['ref_image_sub'].to(self.device)
        self.ref_kspace_mask2d = data['ref_kspace_mask2d'].to(self.device)

        self.tag_kspace_full = data['tag_kspace_full'].to(self.device)
        self.tag_kspace_sub = data['tag_kspace_sub'].to(self.device)
        self.tag_image_full = data['tag_image_full'].to(self.device)
        self.tag_image_sub = data['tag_image_sub'].to(self.device)
        self.tag_kspace_mask2d = data['tag_kspace_mask2d'].to(self.device)

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        I = self.tag_image_sub
        I.requires_grad_(True)

        net = {}
        for i in range(1, self.n_recurrent + 1):
            '''Image Space'''
            if self.opts.use_prior:
                x_I = torch.cat((I, self.ref_image_full), 1)
            else:
                x_I = I

            net['r%d_img_pred' % i] = self.net_G_I(x_I)  # output recon image
            net['r%d_img_dc_pred' % i], _ = self.dcs_I[i - 1](net['r%d_img_pred' % i], self.tag_kspace_full, self.tag_kspace_mask2d)

            '''K Space'''
            net['r%d_kspc_img_dc_pred' % i] = fft2(net['r%d_img_dc_pred' % i].permute(0, 2, 3, 1))  # output data consistency image's kspace

            if self.opts.use_prior:
                x_K = torch.cat((net['r%d_kspc_img_dc_pred' % i].permute(0, 3, 1, 2), self.ref_kspace_full.permute(0, 3, 1, 2)), 1)  # concatenate reference kspace fully sampled
            else:
                x_K = net['r%d_kspc_img_dc_pred' % i].permute(0, 3, 1, 2)

            net['r%d_kspc_pred' % i] = self.net_G_K(x_K)  # output recon kspace
            I, _ = self.dcs_K[i - 1](net['r%d_kspc_pred' % i], self.tag_kspace_full, self.tag_kspace_mask2d)  # output data consistency images

        self.net = net
        self.recon = I

    def update_G(self):
        loss_G_L1 = 0
        self.optimizer_G.zero_grad()

        # Image domain
        loss_img_dc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_img_dc = loss_img_dc + self.criterion(self.net['r%d_img_dc_pred' % j], self.tag_image_full)

        # Kspace domain
        loss_kspc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_kspc = loss_kspc + self.criterion(self.net['r%d_kspc_pred' % j].permute(0, 2, 3, 1), self.tag_kspace_full)

        loss_G_L1 = loss_img_dc + loss_kspc
        self.loss_G_L1 = loss_G_L1.item()
        self.loss_img = loss_img_dc.item()
        self.loss_kspc = loss_kspc.item()

        total_loss = loss_G_L1
        total_loss.backward()
        self.optimizer_G.step()

    def optimize(self):
        self.loss_G_L1 = 0

        self.forward()
        self.update_G()

    @property
    def loss_summary(self):
        message = ''
        if self.opts.wr_L1 > 0:
            message += 'G_L1: {:.4e} Img_L1: {:.4e} Kspc_L1: {:.4e}'.format(self.loss_G_L1, self.loss_img, self.loss_kspc)

        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))

    def save(self, filename, epoch, total_iter):

        state = {}
        if self.opts.wr_L1 > 0:
            state['net_G_I'] = self.net_G_I.module.state_dict()
            state['net_G_K'] = self.net_G_K.module.state_dict()
            state['opt_G'] = self.optimizer_G.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)

        if self.opts.wr_L1 > 0:
            self.net_G_I.module.load_state_dict(checkpoint['net_G_I'])
            self.net_G_K.module.load_state_dict(checkpoint['net_G_K'])
            if train:
                self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    def evaluate(self, loader):
        val_bar = tqdm(loader)
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()

        recon_images = []
        gt_images = []
        input_images = []

        for data in val_bar:
            self.set_input(data)
            self.forward()

            if self.opts.wr_L1 > 0:
                psnr_recon = psnr(complex_abs_eval(self.recon),
                                  complex_abs_eval(self.tag_image_full))
                avg_psnr.update(psnr_recon)

                ssim_recon = ssim(complex_abs_eval(self.recon)[0,0,:,:].cpu().numpy(),
                                  complex_abs_eval(self.tag_image_full)[0,0,:,:].cpu().numpy())
                avg_ssim.update(ssim_recon)

                recon_images.append(self.recon[0].cpu())
                gt_images.append(self.tag_image_full[0].cpu())
                input_images.append(self.tag_image_sub[0].cpu())

            message = 'PSNR: {:4f} '.format(avg_psnr.avg)
            message += 'SSIM: {:4f} '.format(avg_ssim.avg)
            val_bar.set_description(desc=message)

        self.psnr_recon = avg_psnr.avg
        self.ssim_recon = avg_ssim.avg

        self.results = {}
        if self.opts.wr_L1 > 0:
            self.results['recon'] = torch.stack(recon_images).squeeze().numpy()
            self.results['gt'] = torch.stack(gt_images).squeeze().numpy()
            self.results['input'] = torch.stack(input_images).squeeze().numpy()