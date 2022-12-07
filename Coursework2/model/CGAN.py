import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST

from classifier.model import Classifier

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

class SyntheticMNIST(Dataset):
    def __init__(self, synthetic):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.mnist      = MNIST(os.getcwd(), train=True, download=True, transform=transform)        
        self.size       = 60000
        self.synthetic  = synthetic
        self.syn_imgs   = torch.load('./synthetic_dataset/synthetic_60k_images.pt')
        self.syn_labs   = torch.load('./synthetic_dataset/synthetic_60k_labels.pt')

        num_synth   = int(self.size * synthetic)
        num_real    = self.size - num_synth
        idx_perm    = torch.randperm(self.size)
        sidx        = idx_perm[:num_synth]

        # bitmap indicating if idx is synthetic (1) or not (0)
        self.synth_bm       = torch.zeros(self.size) 
        self.synth_bm[sidx] = 1

    def __getitem__(self,i):
        if self.synth_bm[i]:
            return tuple([self.syn_imgs[i], int(self.syn_labs[i])])
        return self.mnist[i]

    def __len__(self):
        return self.size


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, n_classes):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.output_dim = output_dim
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), self.output_dim)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(input_dim)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

class DiscriminatorLabel(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(input_dim)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, n_classes),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

class GAN(LightningModule):

    def __init__(self, cfg, wandb):
        super().__init__()
        self.save_hyperparameters()

        # hyperparams
        self.latent_dim = cfg['latent_dim']
        self.lr         = cfg['lr']
        self.b1         = cfg['b1']
        self.b2         = cfg['b2']
        self.batch_size = cfg['batch_size']
        self.n_classes  = cfg['n_classes']

        self.wandb      = wandb
       
        self.img_shape  = cfg['mnist_shape']
        self.img_dim    = np.prod(cfg['mnist_shape']) # (1, 28, 28)

        # data
        self.train_data = SyntheticMNIST(cfg['synthetic'])
        
        # networks
        self.generator      = Generator(input_dim=self.latent_dim, output_dim=self.img_dim, n_classes=self.n_classes)
        self.discriminator  = Discriminator(self.img_dim, n_classes=self.n_classes)
  
        self.validation_z = torch.randn(8, self.latent_dim)
        self.validation_label = torch.randint(0, self.n_classes, (8,))
        
        self.example_input_array = (
            torch.zeros(2, self.latent_dim),
            torch.LongTensor(np.random.randint(0, self.n_classes, 2)))

        # ideal classifier
        clf_weight_path = './classifier/weight.ckpt'
        assert(os.path.exists(clf_weight_path))
        self.clf        = Classifier.load_from_checkpoint(clf_weight_path, lr_rate=1e-3)

        # inceptions metrics
        self.FID    = FrechetInceptionDistance(normalize=True)
        self.IS     = InceptionScore(normalize=True)
        self.KID    = KernelInceptionDistance(subset_size=50, normalize=True)

    def forward(self, z, label):
        return self.generator(z, label)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)
        
        # fake label
        gen_labels = torch.randint(0, self.n_classes, (imgs.shape[0],))
        gen_labels = gen_labels.type_as(labels)
        
        # generate fake images
        fake_image = self(z, gen_labels)
            
        # train generator
        if optimizer_idx == 0:
            
            # log sampled images
            if batch_idx % 400 == 0 and self.wandb is not None:
                sample_imgs = list(fake_image[:6].clone().detach())
                sample_imgs = list(map(lambda x: x.reshape(28,28), sample_imgs))

                sample_labl = gen_labels[:6].tolist()

                self.wandb.log_image(key='training image',
                                     images=sample_imgs,
                                     caption=sample_labl)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            pred_fake = self.discriminator(fake_image, gen_labels)
            g_loss = self.adversarial_loss(pred_fake, valid)
        
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            self.log('g_loss', g_loss)

            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            # real loss
            pred_real = self.discriminator(imgs, labels)
            real_loss = self.adversarial_loss(pred_real, valid)

            # fake loss
            pred_fake = self.discriminator(fake_image.detach(), gen_labels)
            fake_loss = self.adversarial_loss(pred_fake, fake)

            # discriminator loss is the average of these
            d_loss = 0.5 * (real_loss + fake_loss)
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            self.log('d_loss', d_loss)

            return output


    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def on_train_epoch_end(self):
        z     = self.validation_z.to(self.device)
        label = self.validation_label.to(self.device)
        
        # log sampled images
        sample_imgs = list(self(z, label).clone().detach())
        sample_imgs = list(map(lambda x: x.reshape(28,28), sample_imgs))
        sample_labl = label.tolist()

        self.wandb.log_image(key='validation image', 
                             images=sample_imgs,
                             caption=sample_labl)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size)

    def on_validation_epoch_start(self):
        self.IS.reset()
        self.FID.reset()
        self.KID.reset()

    def validation_step(self, batch, batch_idx):
        reals, real_labels  = batch
        batch_size          = reals.shape[0]

        with torch.no_grad():
            z       = torch.randn(batch_size, self.latent_dim)
            z       = z.type_as(reals)
            fakes   = self(z, real_labels)

            ########## Inceptions ##########
            fake_imgs = fakes.reshape(batch_size,1,28,28).expand(-1,3,-1,-1).type(torch.uint8)
            real_imgs = reals.expand(-1, 3, -1, -1).type(torch.uint8)

            self.IS.update(fake_imgs)
            self.FID.update(fake_imgs, real=False)
            self.FID.update(real_imgs, real=True)
            self.KID.update(fake_imgs, real=False)
            self.KID.update(real_imgs, real=True)
            ################################

            ########## TARR ##########
            fake_logits = self.clf(fakes.reshape(batch_size,1,28,28)) 
            fake_labels = torch.argmax(fake_logits, axis=1)
            
            tarr = (fake_labels == real_labels).float().mean()

            self.log("TARR", tarr, sync_dist=True)
            ##########################

    def on_validation_epoch_end(self):
        is_m, is_s      = self.IS.compute()
        fid             = self.FID.compute()
        kid_m, kid_s    = self.KID.compute()

        self.log("IS_mean", is_m,   sync_dist=True)
        self.log("IS_std",  is_s,   sync_dist=True)
        self.log("FID",     fid,    sync_dist=True)
        self.log("KID_mean",kid_m,  sync_dist=True)
        self.log("KID_std", kid_s,  sync_dist=True)

