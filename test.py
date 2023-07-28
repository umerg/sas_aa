"""
Testing File for P90, Generalised Performance
Usage:
    train.py --config=<config-file>

Options:
    --config=<config-file>   Path to config file containing hyperparameter info
"""
import os
import logging
from docopt import docopt
import torch.utils.data as data
from pytorch_lightning import Trainer, seed_everything

from config import Config
from datamodule import MovieDataModule
from model import LightSAS

import torch

print(torch.backends.mps.is_available())


# set to online to use wandb
#os.environ["WANDB_MODE"] = "online"
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    args = docopt(__doc__)
    config = Config(args["--config"])

    seed_everything(42)

    trainer = Trainer(
        devices=config.devices, 
        accelerator=config.accelerator # If your machine has GPUs, it will use the GPU Accelerator for training
    )
    path = config.checkpoint
    model = LightSAS.load_from_checkpoint(path)

    '''
    #for p90
    logging.info("Testing for full..")
    data_module = MovieDataModule(config, 'full', True)
    trainer.test(model, datamodule = data_module, verbose=True)

    #for front
    logging.info("Testing for front..")
    data_module = MovieDataModule(config, 'front', False)
    trainer.test(model, datamodule = data_module, verbose=True)

    #for mid
    logging.info("Testing for middle..")
    data_module = MovieDataModule(config, 'mid', False)
    trainer.test(model, datamodule = data_module, verbose=True)

    #for back
    logging.info("Testing for back..")
    data_module = MovieDataModule(config, 'back', False)
    trainer.test(model, datamodule = data_module, verbose=True)
    '''

    #for back
    logging.info("Testing for back wo last..")
    data_module = MovieDataModule(config, 'bwol', False)
    trainer.test(model, datamodule = data_module, verbose=True)

