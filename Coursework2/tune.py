import os
from argparse import ArgumentParser
from importlib import import_module

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from Config import Config
from model.CGAN import GAN as CGAN
from model.DCGAN import GAN as DCGAN

import yaml

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback

def run(opt):

    with open(opt.config, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

    wandb_logger = WandbLogger(project=opt.gan)

    if opt.gan == "CGAN":
        model   = CGAN(cfg, wandb_logger)
    if opt.gan == "DCGAN":
        model   = DCGAN(cfg, wandb_logger)

    assert(os.path.exists("./weights"))
    ckpt_callback = ModelCheckpoint(dirpath="./weights",
                                    filename=opt.gan+"_{epoch}")

    dcgan_tune_callback = TuneReportCallback({
                                "g_loss": "g_loss"
                            }, on="validation_end")

    config = {
        "": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    trainer = Trainer(accelerator="gpu", 
                      devices=opt.ngpu, 
                      strategy="ddp",
                      logger=wandb_logger,
                      callbacks=[ckpt_callback],
                      max_epochs=200,
                      )
    trainer.fit (model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="GAN model name")
    parser.add_argument("--ngpu", type=int, help="number of gpus")
    parser.add_argument("--gan", 
                        type=str, 
                        choices=['CGAN', 'DCGAN'],
                        help="GAN model name")
    opt = parser.parse_args()

    run(opt)
