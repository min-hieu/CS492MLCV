import torch
import pytorch_lightning as pl
import os

from torch import nn
from torch.nn import functional as F


class Classifier(pl.LightningModule):

  def __init__(self, lr_rate):
    super(Classifier, self).__init__()
    
    # mnist images are (1, 28, 28) (channels, width, height) 
    self.layer_1 = torch.nn.Linear(28 * 28, 128)
    self.layer_2 = torch.nn.Linear(128, 256)
    self.layer_3 = torch.nn.Linear(256, 10)
    self.lr_rate = lr_rate

    self.val_losses = []
    self.test_losses = []

  def forward(self, x):
      batch_size, channels, width, height = x.size()
      
      # (b, 1, 28, 28) -> (b, 1*28*28)
      x = x.view(batch_size, -1)

      # layer 1 (b, 1*28*28) -> (b, 128)
      x = self.layer_1(x)
      x = torch.relu(x)

      # layer 2 (b, 128) -> (b, 256)
      x = self.layer_2(x)
      x = torch.relu(x)

      # layer 3 (b, 256) -> (b, 10)
      x = self.layer_3(x)

      # probability distribution over labels
      x = torch.softmax(x, dim=1)

      return x

  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)

  def training_step(self, train_batch, batch_idx):
      x, y = train_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)

      self.log('train_loss', loss)
      return loss

  def validation_step(self, val_batch, batch_idx):
      x, y = val_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      self.log('val_loss', loss)
      self.val_losses.append(loss)

  def test_step(self, val_batch, batch_idx):
      x, y = val_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      self.log('test_loss', loss)
      self.test_losses.append(loss)

  def on_validation_epoch_end(self):
      avg_loss = torch.tensor(self.val_losses).mean()
      self.log('avg_val_loss', avg_loss)
      self.val_losses = []

  def on_test_epoch_end(self):
      avg_loss = torch.tensor(self.test_losses).mean()
      self.log('avg test_loss', avg_loss)
      self.test_losses = []

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
    lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95),
                    'name': 'expo_lr'}
    return [optimizer], [lr_scheduler]
