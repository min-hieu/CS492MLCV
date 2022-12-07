from model import Classifier 
import torch
import pytorch_lightning as pl
import os

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def prepare_data():
  # transforms for images
  transform=transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))])
    
  # prepare transforms standard to MNIST
  mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
  
  mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

  mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

  return mnist_train, mnist_val, mnist_test

train, val, test = prepare_data()
train_loader, val_loader, test_loader = DataLoader(train, batch_size=64), DataLoader(val, batch_size=64), DataLoader(test, batch_size=64)

checkpoint_callback = ModelCheckpoint(dirpath=os.getcwd(),
                                      filename='mnist_clf_{epoch}',
                                      monitor='val_loss', 
                                      mode='min', save_top_k=3)

model = Classifier(lr_rate=1e-3)

wandb_logger = WandbLogger(project="MNIST Classifier")
trainer = pl.Trainer(accelerator="gpu", devices=1,
                     max_epochs=50,
                     logger=wandb_logger,
                     callbacks=[checkpoint_callback],
                     default_root_dir=os.getcwd()) 

trainer.fit(model, train_loader, val_loader)
