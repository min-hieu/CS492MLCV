import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class VirtualBatchNorm(nn.Module):
    r"""Virtual Batch Normalization Module as proposed in the paper
    - :math:`x` : Batch Being Normalized
    - :math:`x_{ref}` : Reference Batch
    Args:
        in_features (int): Size of the input dimension to be normalized
        eps (float, optional): Value to be added to variance for numerical stability while normalizing
    """

    def __init__(self, in_features, eps=1e-5):
        super(VirtualBatchNorm, self).__init__()
        self.in_features    = in_features
        self.scale          = nn.Parameter(torch.ones(in_features))
        self.bias           = nn.Parameter(torch.zeros(in_features))
        self.ref_mu         = None
        self.ref_var        = None
        self.eps            = eps

    def _batch_stats(self, x):
        mu  = torch.mean(x, dim=0, keepdim=True)
        var = torch.var(x,  dim=0, keepdim=True)
        return mu, var

    def _normalize(self, x, mu, var):
        std     = torch.sqrt(self.eps + var)
        x       = (x - mu) / std
        sizes   = list(x.size())
        for dim, i in enumerate(x.size()):
            if dim != 1:
                sizes[dim] = 1
        scale   = self.scale.view(*sizes)
        bias    = self.bias.view(*sizes)
        return x * scale + bias

    def forward(self, x, clear=False):
        assert x.size(1) == self.in_features
        if self.ref_mu is None or self.ref_var is None:
            self.ref_mu, self.ref_var = self._batch_stats(x)
            self.ref_mu     = self.ref_mu.clone().detach()
            self.ref_var    = self.ref_var.clone().detach()

        out = self._normalize(x, self.ref_mu, self.ref_var)

        if clear:
            self.ref_mu     = None
            self.ref_var    = None

        return out

class Generator(nn.Module):
    def __init__(self, input_dim, img_size, channels, vbn):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        if not vbn:
            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, channels, 3, stride=1, padding=1),
                nn.Tanh(),
            )
        else:
            self.conv_blocks = nn.Sequential(
                VirtualBatchNorm(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                VirtualBatchNorm(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                VirtualBatchNorm(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, channels, 3, stride=1, padding=1),
                nn.Tanh(),
            )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)

        return img
    
class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        ds_size = img_size // 2 ** 4
        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            nn.Flatten(),
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


class GAN(LightningModule):

    def __init__(self, cfg, wandb):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = cfg['latent_dim']
        self.lr         = cfg['lr']
        self.b1         = cfg['b1']
        self.b2         = cfg['b2']
        self.batch_size = cfg['batch_size']
        self.n_cpus     = cfg['cpus']
        self.vbn        = cfg['vbn']

        self.wandb      = wandb
        
        self.channels, self.img_width, self.img_height = cfg['mnist_shape']
        self.img_dim = np.prod(cfg['mnist_shape']) # (1, 32, 32)
        
        # networks
        self.generator = Generator(input_dim=self.latent_dim, img_size=self.img_width, 
                                   channels=self.channels, vbn=self.vbn)
        self.discriminator = Discriminator(img_size=self.img_width, channels=self.channels)
    
        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
  
        self.validation_z = torch.randn(64, self.latent_dim)
        self.example_input_array = torch.randn(2, self.latent_dim)

        # inceptions metrics 
        self.IS     = InceptionScore(normalize=True)
        self.FID    = FrechetInceptionDistance(normalize=True)
        self.KID    = KernelInceptionDistance(subset_size=50, normalize=True)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)
        
        # generate fake images
        fake_image = self(z)
            
        # train generator
        if optimizer_idx == 0:
        
            # log sampled images
            if batch_idx % 400 == 0:
                sample_imgs = list(fake_image[:6].clone().detach())
                sample_imgs = list(map(lambda x: x.reshape(32,32), sample_imgs))
                self.wandb.log_image(key="training sample", images=sample_imgs)

            # ground truth result
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            pred_fake = self.discriminator(fake_image)
            g_loss = self.adversarial_loss(pred_fake, valid)

            self.log('g_loss', g_loss)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            
            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            # real loss
            pred_real = self.discriminator(imgs)
            real_loss = self.adversarial_loss(pred_real, valid)

            # fake loss
            pred_fake = self.discriminator(fake_image.detach())
            fake_loss = self.adversarial_loss(pred_fake, fake)

            # discriminator loss is the average of these
            d_loss = 0.5 * (real_loss + fake_loss)

            self.log('d_loss', d_loss)

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.img_width, self.img_height)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_cpus)

    def on_train_epoch_end(self):
        z = self.validation_z.to(self.device)

        sample_imgs = list(self(z))
        sample_imgs = list(map(lambda x: x.reshape(32,32), sample_imgs))
        self.wandb.log_image(key="validation sample", images=sample_imgs)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.img_width, self.img_height)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_cpus)

    def on_validation_epoch_start(self):
        self.IS.reset()
        self.FID.reset()
        self.KID.reset()
    
    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        batch_size = imgs.shape[0]

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)
        
        # generate fake images
        fake_image = self(z)

        ############# Generator ##############
        # ground truth result
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        # adversarial loss is binary cross-entropy
        pred_fake = self.discriminator(fake_image)
        g_loss = self.adversarial_loss(pred_fake, valid)

        self.log('val_g_loss', g_loss)
        ######################################

        ############# Discriminator ##############
        # Adversarial ground truths
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        # real loss
        pred_real = self.discriminator(imgs)
        real_loss = self.adversarial_loss(pred_real, valid)

        # fake loss
        pred_fake = self.discriminator(fake_image.detach())
        fake_loss = self.adversarial_loss(pred_fake, fake)

        # discriminator loss is the average of these
        d_loss = 0.5 * (real_loss + fake_loss)

        self.log('val_d_loss', d_loss)
        ##########################################

        ############# Inceptions ##############
        fake_imgs = fake_image.reshape(batch_size,1,32,32).expand(-1,3,-1,-1)
        real_imgs = imgs.expand(-1,3,-1,-1)

        self.IS.update(fake_imgs)
        self.FID.update(fake_imgs, real=False)
        self.FID.update(real_imgs, real=True)
        self.KID.update(fake_imgs, real=False)
        self.KID.update(real_imgs, real=True)
        #######################################

    def on_validation_epoch_end(self):
        is_m, is_s      = self.IS.compute()
        fid             = self.FID.compute()
        kid_m, kid_s    = self.KID.compute()

        self.log("IS_mean", is_m,   sync_dist=True)
        self.log("IS_std",  is_s,   sync_dist=True)
        self.log("FID",     fid,    sync_dist=True)
        self.log("KID_mean",kid_m,  sync_dist=True)
        self.log("KID_std", kid_s,  sync_dist=True)

