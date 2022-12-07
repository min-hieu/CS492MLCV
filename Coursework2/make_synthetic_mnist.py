import os, torch, yaml
from model.CGAN import GAN
from multiprocessing import Pool
from tqdm import tqdm 
from itertools import product
import matplotlib.pyplot as plt
import numpy as np

with open('./config/CGAN.yml') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

weight_path = './weights/cgan.ckpt'
cgan        = GAN.load_from_checkpoint(weight_path,
                                       cfg=cfg,
                                       wandb=None)

with torch.no_grad():
    datasize    = 60000
    syn_z       = torch.randn(datasize, cfg['latent_dim'])
    syn_labels  = torch.randint(10, [datasize])
    syn_imgs    = cgan(syn_z, syn_labels).reshape(-1, 1, 28, 28)

    torch.save(syn_imgs,   './synthetic_dataset/synthetic_60k_images.pt')
    torch.save(syn_labels, './synthetic_dataset/synthetic_60k_labels.pt')

    # visualization

    params = {
        'axes.titlesize': 8,
        'axes.titleweight':'100',
        'figure.titlesize': 12,
        'figure.titleweight':'600',
    }

    # Updating the rcParams in Matplotlib
    plt.rcParams.update(params)

    sample_dim = (8,16)
    sample_idx = torch.randperm(datasize)[:np.prod(sample_dim)]
    sample_syn_imgs = syn_imgs[sample_idx].reshape(-1, 28, 28)
    sample_syn_labs = syn_labels[sample_idx]
    fig, axes = plt.subplots(*sample_dim, gridspec_kw={'wspace':0, 'hspace':0},
                             squeeze=True)
    fig.suptitle('sample synthetic dataset', y=0.94)
    for (i,j) in tqdm(product(range(sample_dim[0]), range(sample_dim[1])), desc="plotting"):
        idx = i * sample_dim[1] + j
        samp_img = sample_syn_imgs[idx]
        axes[i,j].set_title(f"{sample_syn_labs[idx]}", pad=3)
        axes[i,j].axis("off")
        axes[i,j].imshow(samp_img)

    fig.savefig("./synthetic_dataset/preview.png", dpi=400)
