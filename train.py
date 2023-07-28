"""
Training File
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
from pytorch_lightning.callbacks import ModelCheckpoint


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
    
    data_module = MovieDataModule(config, 'full', False)

    
    logging.info("Setting up the model..")
    model = LightSAS(config)

    checkpoint_callback = ModelCheckpoint(dirpath = "./checkpoints/" + config.run_name, save_top_k = 3, monitor = "val/ndcg", mode = "max")

    trainer = Trainer(
        fast_dev_run=False,
        log_every_n_steps=5,
        check_val_every_n_epoch=5, 
        callbacks = [checkpoint_callback],
        devices=config.devices, 
        max_epochs=config.epochs,
        accelerator=config.accelerator # If your machine has GPUs, it will use the GPU Accelerator for training
    )
    logging.info("Training the model..")
    trainer.fit(model, data_module)

    logging.info("Testing the model..")
    trainer.test(datamodule = data_module, ckpt_path="best", verbose=True)
