#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

from utils import load_hdr_as_tensor

import os
import cv2
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw
import OpenEXR

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def load_testset(root_dir, params, shuffled=False, single=False):
    """Loads test set and returns corresponding data loader."""
    noise = ('gaussian', 50)

    dataset = NoisyDataset(root_dir, params.redux, params.crop_size, rgb=params.rgb, transform=params.transform,
        clean_targets=params.clean_targets, noise_dist=noise, seed=params.seed, test=True)

    # Use batch size of 1, if requested (e.g. test set)
    # if len(params.indices)==1:
    return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    # else:
    #     return DataLoader(dataset, batch_size=8, shuffle=shuffled)


def load_dataset(root_dir, redux, params, shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""

    # Create Torch dataset
    noise = (params.noise_type, params.noise_param)

    # Instantiate appropriate dataset class
    if params.noise_type == 'mc':
        dataset = MonteCarloDataset(root_dir, redux, params.crop_size,
            clean_targets=params.clean_targets)
    else:
        dataset = NoisyDataset(root_dir, redux, params.crop_size, rgb=params.rgb, transform=params.transform,
            clean_targets=params.clean_targets, noise_dist=noise, seed=params.seed)

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128, clean_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.clean_targets = clean_targets

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs


    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)


class NoisyDataset(AbstractDataset):
    """Class for injecting random noise into dataset."""

    def __init__(self, root_dir, redux, crop_size, rgb, transform, clean_targets=False,
        noise_dist=('gaussian', 50.), seed=None, test=False):
        """Initializes noisy image dataset."""

        super(NoisyDataset, self).__init__(root_dir, redux, crop_size, clean_targets)
        
        mode = root_dir.split('/')[-1]
        self.clean_img_dir = os.path.join(root_dir, 'clean')
        self.noisy_img_dir = os.path.join(root_dir, 'noisy')
        
        self.imgs = os.listdir(root_dir+'/clean') #can be noisy as well, since file names are same

        if redux:
            self.imgs = self.imgs[:redux]
        # elif test:
        #     indices = [int(i) for i in indices]
        #     self.imgs = list(np.array(self.imgs)[indices])


        #self.noise_for_source = [None]*len(self.imgs) #list of constant noise for source across all epochs specific to each image
        # Noise parameters (max std for Gaussian, lambda for Poisson, nb of artifacts for text)
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)

        self.rgb = rgb
        self.transform = transform


    def _add_noise(self, img, sigma=None):
        """Adds Gaussian or Poisson noise to image."""
        if len(img.shape)==3:
            w, h, c = img.shape
        else:
          w, h = img.shape
          c = 1
        # w, h = img.size
        # c = len(img.getbands())
        #print(c, w, h)
        # Poisson distribution
        # It is unclear how the paper handles this. Poisson noise is not additive,
        # it is data dependent, meaning that adding sampled valued from a Poisson
        # will change the image intensity...
        if self.noise_type == 'poisson':
            noise = np.random.poisson(img)
            noise_img = img + noise
            noise_img = 255 * (noise_img / np.amax(noise_img))

        # Normal distribution (default)
        else:
            if sigma:
              std = sigma
            else:
              std = self.noise_param

            if c==1:
              noise = np.random.normal(0, std, (h, w))
            else:
              noise = np.random.normal(0, std, (h, w, c))
            # Add noise and clip
            noise_img = np.array(img) + noise
        # print('noise ', noise.shape)
        # print('img ', np.array(img).shape)
        # print('noise_img ', noise_img.shape)
        #noise_img = np.clip(noise_img, 0, 1).astype(np.float32)
        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)

        #return Image.fromarray(noise_img)
        return noise_img


    def _corrupt(self, img, sigma=None):
        """Corrupts images (Gaussian, Poisson, or text overlay)."""

        if self.noise_type in ['gaussian', 'poisson']:
            return self._add_noise(img, sigma)
        elif self.noise_type == 'text':
            return self._add_text_overlay(img)
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type))


    def anscombe(self, image):
        '''
        Compute the anscombe variance stabilizing transform.
        the input   x   is noisy Poisson-distributed data
        the output  fx  has variance approximately equal to 1.
        Reference: Anscombe, F. J. (1948), "The transformation of Poisson,
        binomial and negative-binomial data", Biometrika 35 (3-4): 246-254
        '''
        return 2.0*np.sqrt(image + 3.0/8.0)


    def log(self, image):
        c = 100 / np.log(1 + np.max(image))
        log_image = c*(np.log(image + 1))
        log_image = np.array(log_image, dtype=np.uint8)
        return c, log_image


    def normalize(self, image):
        if isinstance(image, torch.Tensor):
            image = (image-image.min())/(image.max()-image.min())
        else:
            image = (image-np.min(image))/(np.max(image)-np.min(image))
            image = Image.fromarray(image)
        return image
    
    
    def rescale(self, image):
        image = (image-np.min(image))/(np.max(image)-np.min(image))
        #image = Image.fromarray(image*255)
        image = image*255
        return image


    def to_tensor(self, image):
      return torch.tensor(np.array(image)).unsqueeze(0)


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        clean_path = os.path.join(self.clean_img_dir, self.imgs[index])
        noisy_path = os.path.join(self.noisy_img_dir, self.imgs[index])
        if self.rgb:
          clean = Image.open(clean_path).convert('RGB')
          noisy = Image.open(noisy_path).convert('RGB')
        else:
          # clean = Image.open(clean_path).convert('L')
          # noisy = Image.open(noisy_path).convert('L')
          clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
          noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)

        # Clean image
        clean          = self.to_tensor(clean)
        noisy_original = self.to_tensor(noisy)

        # Transforming noisy image
        if self.transform=="anscombe":
            #noisy_transformed = Image.fromarray(self.anscombe(np.array(noisy)).astype('float32')) 
            noisy_transformed = self.anscombe(np.array(noisy).astype('float32')) 
        else:   #for none
            noisy_transformed = noisy
          
        

        #Rescale from 0-255

        noisy_transformed = self.rescale(noisy_transformed)

        source = self.to_tensor(self._corrupt(self._corrupt(noisy_transformed)))
        target = self.to_tensor(self._corrupt(noisy_transformed))  
        noisy_transformed = self.to_tensor(noisy_transformed)     
        
        # print("source", source.max(), source.min())
        # print("target", target.max(), target.min())
        # print("clean", clean.max(), clean.min())
        # print("noisy_transformed", noisy_transformed.max(), noisy_transformed.min())
        # print("noisy_original", noisy_original.max(), noisy_original.min())
        # print()

        clean = self.normalize(clean)
        source = self.normalize(source)
        target = self.normalize(target)
        noisy_original = self.normalize(noisy_original)
        noisy_transformed = self.normalize(noisy_transformed)
        
        if self.transform=="log":
            return source, target, clean, [torch.tensor(c), noisy_transformed], noisy_original
        else:
            return source, target, clean, noisy_transformed, noisy_original


    def _add_text_overlay(self, img):
        """Adds text overlay to images."""

        assert self.noise_param < 1, 'Text parameter is an occupancy probability'

        w, h = img.size
        c = len(img.getbands())

        # Choose font and get ready to draw
        if platform == 'linux':
            serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        else:
            serif = 'Times New Roman.ttf'
        text_img = img.copy()
        text_draw = ImageDraw.Draw(text_img)

        # Text binary mask to compute occupancy efficiently
        w, h = img.size
        mask_img = Image.new('1', (w, h))
        mask_draw = ImageDraw.Draw(mask_img)

        # Random occupancy in range [0, p]
        if self.seed:
            random.seed(self.seed)
            max_occupancy = self.noise_param
        else:
            max_occupancy = np.random.uniform(0, self.noise_param)
        def get_occupancy(x):
            y = np.array(x, dtype=np.uint8)
            return np.sum(y) / y.size

        # Add text overlay by choosing random text, length, color and position
        while 1:
            font = ImageFont.truetype(serif, np.random.randint(16, 21))
            length = np.random.randint(10, 25)
            chars = ''.join(random.choice(ascii_letters) for i in range(length))
            color = tuple(np.random.randint(0, 255, c))
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            text_draw.text(pos, chars, color, font=font)

            # Update mask and check occupancy
            mask_draw.text(pos, chars, 1, font=font)
            if get_occupancy(mask_img) > max_occupancy:
                break

        return text_img

class MonteCarloDataset(AbstractDataset):
    """Class for dealing with Monte Carlo rendered images."""

    def __init__(self, root_dir, redux, crop_size,
        hdr_buffers=False, hdr_targets=True, clean_targets=False):
        """Initializes Monte Carlo image dataset."""

        super(MonteCarloDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        # Rendered images directories
        self.root_dir = root_dir
        self.imgs = os.listdir(os.path.join(root_dir, 'render'))
        self.albedos = os.listdir(os.path.join(root_dir, 'albedo'))
        self.normals = os.listdir(os.path.join(root_dir, 'normal'))

        if redux:
            self.imgs = self.imgs[:redux]
            self.albedos = self.albedos[:redux]
            self.normals = self.normals[:redux]

        # Read reference image (converged target)
        ref_path = os.path.join(root_dir, 'reference.png')
        self.reference = Image.open(ref_path).convert('RGB')

        # High dynamic range images
        self.hdr_buffers = hdr_buffers
        self.hdr_targets = hdr_targets


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Use converged image, if requested
        if self.clean_targets:
            target = self.reference
        else:
            target_fname = self.imgs[index].replace('render', 'target')
            file_ext = '.exr' if self.hdr_targets else '.png'
            target_fname = os.path.splitext(target_fname)[0] + file_ext
            target_path = os.path.join(self.root_dir, 'target', target_fname)
            if self.hdr_targets:
                target = tvF.to_pil_image(load_hdr_as_tensor(target_path))
            else:
                target = Image.open(target_path).convert('RGB')

        # Get buffers
        render_path = os.path.join(self.root_dir, 'render', self.imgs[index])
        albedo_path = os.path.join(self.root_dir, 'albedo', self.albedos[index])
        normal_path =  os.path.join(self.root_dir, 'normal', self.normals[index])

        if self.hdr_buffers:
            render = tvF.to_pil_image(load_hdr_as_tensor(render_path))
            albedo = tvF.to_pil_image(load_hdr_as_tensor(albedo_path))
            normal = tvF.to_pil_image(load_hdr_as_tensor(normal_path))
        else:
            render = Image.open(render_path).convert('RGB')
            albedo = Image.open(albedo_path).convert('RGB')
            normal = Image.open(normal_path).convert('RGB')

        # Crop
        if self.crop_size != 0:
            buffers = [render, albedo, normal, target]
            buffers = [tvF.to_tensor(b) for b in self._random_crop(buffers)]

        # Stack buffers to create input volume
        source = torch.cat(buffers[:3], dim=0)
        target = buffers[3]

        return source, target
