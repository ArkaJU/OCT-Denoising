#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torchvision.utils import save_image

from unet import UNet
from utils import *
from ssim_metric2 import ssim
from ssim_metric_scratch import ssim_scratch

import os
import json


class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.trainable = trainable
        self.best_valid_loss = float('inf')
        self.early_stopping = EarlyStopping(patience=10)
        self._compile()


    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        print('Noise2Noise: Learning Image Restoration without Clean Data (Lethinen et al., 2018)')

        # Model (3x3=9 channels for Monte Carlo since it uses 3 HDR buffers)

        self.is_mc = False
        if self.p.rgb:
          self.model = UNet(in_channels=3, out_channels=3)
        else:
          self.model = UNet(in_channels=1, out_channels=1)

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            # Learning rate adjustment
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                patience=self.p.nb_epochs/4, factor=0.5, verbose=True)

            # Loss function
            if self.p.loss == 'hdr':
                assert self.is_mc, 'Using HDR loss on non Monte Carlo images'
                self.loss = HDRLoss()
            elif self.p.loss == 'l2':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()


    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()


    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            if self.p.clean_targets:
                self.save_dir = f'supervised-lr{self.p.learning_rate}-{self.p.noise_param}'
            else:
                self.save_dir = f'self-lr{self.p.learning_rate}-sigma{self.p.noise_param}'
            if self.p.ckpt_overwrite:
                if self.p.clean_targets:
                    self.save_dir = f'clean-lr{self.p.learning_rate}-sigma{self.p.noise_param}'
                else:
                    self.save_dir = self.p.noise_type

            self.save_dir = os.path.join('/content', self.save_dir)
            self.ckpt_dir = os.path.join(self.save_dir, 'ckpts')
            self.results_dir = os.path.join(self.save_dir, 'results')
            
            if not os.path.isdir(self.save_dir):
                os.mkdir(self.save_dir)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/n2n-{}.pt'.format(self.ckpt_dir, self.p.noise_type)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/n2n-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = '{}/n2n-stats.json'.format(self.ckpt_dir)

        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)


    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))


    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""

        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr, valid_ssim, valid_ssim2 = self.eval(valid_loader, epoch)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr, valid_ssim, valid_ssim2)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        stats['valid_ssim'].append(valid_ssim)
        stats['valid_ssim2'].append(valid_ssim2)

        

        if self.best_valid_loss > valid_loss:
            print("Best model so far on the basis of val loss! Saving model.....")
            self.best_valid_loss = valid_loss
            self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if self.p.plot_stats:
            loss_str = f'{self.p.loss.upper()} loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')
            plot_per_epoch(self.ckpt_dir, 'Valid SSIM', stats['valid_ssim'], 'SSIM')
        
        if self.early_stopping(valid_loss):
            return {"stop": True}
        else:
          return {"stop": False}

    def test(self, test_loader, show):
        """Evaluates denoiser on test set."""

        self.model.train(False)

        source_imgs = []
        denoised_imgs = []
        clean_imgs = []

        # Create directory for denoised images
        save_path = os.path.join('/content', 'test_results')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for batch_idx, (_, _, clean, noisy_transformed, noisy_original) in enumerate(test_loader):
            # Only do first <show> images

            if self.p.transform=="log":
                c, noisy_transformed = noisy_transformed

            if self.use_cuda:
                noisy_transformed = noisy_transformed.cuda()
                if self.p.transform=="log":
                    c = c.cuda()
                clean = clean.cuda()
                noisy_original = noisy_original.cuda()

            # Denoise
            noisy_denoised = self.model(noisy_transformed)

            # Inverse transformation
            if self.p.transform=="anscombe":
                noisy_denoised = self.normalize(self.inverse_anscombe(noisy_denoised*255))
            elif self.p.transform=="log":
                noisy_denoised = self.normalize(self.inverse_log(c, noisy_denoised*255))
              

        clean = clean.cpu()
        noisy_original = noisy_original.cpu()
        noisy_denoised = noisy_denoised.cpu()

        # Create montage and save images
        print('Saving images and montages to: {}'.format(save_path))
        for i in range(noisy_denoised.shape[0]):

            img_name = test_loader.dataset.imgs[i]
            save_path_img = os.path.join(save_path, 'idx'+str(i))
            if not os.path.isdir(save_path_img):
                os.mkdir(save_path_img)
            
            print(f"Index: {i}")

            psnr_ = psnr(noisy_denoised[i].cpu(), clean[i].cpu()).item()
            print("PSNR(dB):", psnr_)

            ssim_ = ssim(noisy_denoised[i].unsqueeze(0), clean[i].unsqueeze(0)).item()
            print("SSIM:", ssim_)
            print()

            stats = {"PSNR": psnr_, "SSIM": ssim_}
            fname_dict = '{}/stats.json'.format(save_path_img)
            with open(fname_dict, 'w') as fp:
                json.dump(stats, fp, indent=2)

            save_image(clean[i], os.path.join(save_path_img, f'clean.png'))
            save_image(noisy_denoised[i], os.path.join(save_path_img, f'noisy_denoised.png'))
            save_image(noisy_original[i], os.path.join(save_path_img, f'noisy_original.png'))
            #create_montage(img_name, self.p.noise_type, save_path_img, noisy_original[i], noisy_denoised[i], clean[i], show)

    def inverse_anscombe(self, z):
        '''
        Compute the inverse transform using an approximation of the exact
        unbiased inverse.
        Reference: Makitalo, M., & Foi, A. (2011). A closed-form
        approximation of the exact unbiased inverse of the Anscombe
        variance-stabilizing transformation. Image Processing.
        '''
        #return (z/2.0)**2 - 3.0/8.0
        return (1.0/4.0 * torch.pow(z, 2) +
                1.0/4.0 * torch.sqrt(torch.tensor(3.0/2.0)) * torch.pow(z, -1.0) -
                11.0/8.0 * torch.pow(z, -2.0) + 
                5.0/8.0 * torch.sqrt(torch.tensor(3.0/2.0)) * torch.pow(z, -3.0) - 1.0 / 8.0)


    def normalize(self, image):
        if isinstance(image, torch.Tensor):
            image = (image-image.min())/(image.max()-image.min())
        else:
            image = (image-np.min(image))/(np.max(image)-np.min(image))
            image = Image.fromarray(image)
        return image


    def inverse_log(self, c, image):
        c = c.view(c.shape[0],1,1,1)
        return torch.exp(image*1/c)


    def eval(self, valid_loader, epoch):
        """Evaluates denoiser on validation set."""

        if self.p.clean_targets:
            self.save_dir = f'supervised-lr{self.p.learning_rate}-{self.p.noise_param}'
        else:
            self.save_dir = f'self-lr{self.p.learning_rate}-sigma{self.p.noise_param}'


        self.save_dir = os.path.join('/content', self.save_dir)
        results_path = os.path.join(self.save_dir, 'results')
        
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
    
        if not os.path.isdir(results_path):
            os.mkdir(results_path)


        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()
        ssim_meter = AvgMeter()
        ssim_meter2 = AvgMeter()

        for batch_idx, (source, target, clean, noisy_transformed, noisy_original) in enumerate(valid_loader):
            
            if self.p.transform=="log":
                c, noisy_transformed = noisy_transformed

            if self.use_cuda:
                source = source.cuda()
                target = target.cuda()
                noisy_transformed = noisy_transformed.cuda()
                if self.p.transform=="log":
                    c = c.cuda()
                clean = clean.cuda()
                noisy_original = noisy_original.cuda()

            # Denoise
            noisy_denoised = self.model(noisy_transformed)

            # Inverse transformation
            if self.p.transform=="anscombe":
                noisy_denoised = self.normalize(self.inverse_anscombe(noisy_denoised*255))
            elif self.p.transform=="log":
                noisy_denoised = self.normalize(self.inverse_log(c, noisy_denoised*255))
            
            source_denoised = self.model(source)
            noisy_original_denoised = self.model(noisy_original)

            # Update loss
            loss = self.loss(noisy_denoised, clean)
            loss_meter.update(loss.item())

            # print(clean.dtype)
            # print(noisy_denoised.dtype)
            # print(noisy_transformed.dtype)

            # Compute PSNR
            # if self.is_mc:
            #     source_denoised = reinhard_tonemap(source_denoised)
            # TODO: Find a way to offload to GPU, and deal with uneven batch sizes
            for i in range(self.p.batch_size):
                noisy_denoised = noisy_denoised.cpu()
                noisy_original_denoised = noisy_original_denoised.cpu()
                clean = clean.cpu()
                psnr_meter.update(psnr(noisy_denoised[i], clean[i]).item())
                ssim_meter.update(ssim(noisy_denoised[i].unsqueeze(0), clean[i].unsqueeze(0)).item())
                ssim_meter2.update(ssim_scratch(noisy_denoised[i].unsqueeze(0), clean[i].unsqueeze(0)).item())

            if (batch_idx + 1) % self.p.report_interval//5 == 0 and batch_idx and epoch%10==0:
                epoch_results_path = os.path.join(results_path, f'epoch_{epoch}')
                if not os.path.exists(epoch_results_path):
                  os.makedirs(epoch_results_path)

                epoch_results_path = os.path.join(epoch_results_path, f'val_image_{batch_idx}')
                if not os.path.exists(epoch_results_path):
                  os.makedirs(epoch_results_path)

                save_image(source, os.path.join(epoch_results_path, f'it{batch_idx}_source.png'))
                save_image(source_denoised, os.path.join(epoch_results_path, f'it{batch_idx}_source_denoised.png'))
                save_image(target, os.path.join(epoch_results_path, f'it{batch_idx}_target.png'))
                save_image(clean, os.path.join(epoch_results_path, f'it{batch_idx}_clean.png'))
                save_image(noisy_transformed, os.path.join(epoch_results_path, f'it{batch_idx}_noisy_transformed.png'))
                save_image(noisy_denoised, os.path.join(epoch_results_path, f'it{batch_idx}_noisy_denoised.png'))

                save_image(noisy_original, os.path.join(epoch_results_path, f'it{batch_idx}_noisy_original.png'))
                save_image(noisy_original_denoised, os.path.join(epoch_results_path, f'it{batch_idx}_noisy_original_denoised.png'))


        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = psnr_meter.avg
        ssim_avg = ssim_meter.avg
        ssim_avg2 = ssim_meter2.avg

        return valid_loss, valid_time, psnr_avg, ssim_avg, ssim_avg2


    def train(self, train_loader, valid_loader):
        """Trains denoiser on training set."""

        self.model.train(True)

        self._print_params()
        num_batches = len(train_loader)
        print(num_batches, self.p.report_interval )
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'

        # Dictionaries of tracked stats
        stats = {'noise_type': self.p.noise_type,
                 'noise_param': self.p.noise_param,
                 'train_loss': [],
                 'valid_loss': [],
                 'valid_psnr': [],
                 'valid_ssim': [],
                 'valid_ssim2': []}

        # Main training loop
        
        train_start = datetime.now()
        for epoch in range(self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx, (source, target, clean, noisy_transformed, _) in enumerate(train_loader):
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

                if self.use_cuda:
                    if self.p.clean_targets:
                        clean = clean.cuda()
                        if self.p.transform=="log":
                            noisy_transformed = noisy_transformed[1].cuda()  #no need for c during training
                        else:
                            noisy_transformed = noisy_transformed.cuda()
                    else:
                        source = source.cuda()
                        target = target.cuda()
                      
                # Denoise image
                if self.p.clean_targets:   # for supervised training
                    source_denoised = self.model(noisy_transformed)
                    loss = self.loss(source_denoised, clean)
                else:
                    source_denoised = self.model(source)
                    loss = self.loss(source_denoised, target)

                loss_meter.update(loss.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()

            # Epoch end, save and reset tracker
            early_stop = self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            if early_stop["stop"]:
                print("EarlyStop patience overflow. Stopping training.....")
                break
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))


class HDRLoss(nn.Module):
    """High dynamic range loss."""

    def __init__(self, eps=0.01):
        """Initializes loss with numerical stability epsilon."""

        super(HDRLoss, self).__init__()
        self._eps = eps


    def forward(self, denoised, target):
        """Computes loss by unpacking render buffer."""

        loss = ((denoised - target) ** 2) / (denoised + self._eps) ** 2
        return torch.mean(loss.view(-1))
