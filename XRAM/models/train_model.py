import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from typing import Union, Type

import torch
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

from RAM.trainer import Trainer
from XRAM.data.dataset import get_dataloader

class Config(object):
      
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

config = pd.read_json("XRAM/models/config.json", orient='index', typ='series')
config = Config(config.to_dict())

# Data paths default values
RAW_DATA_PATH = 'data/raw/'
PROCESSED_DATA_PATH = 'data/processed/'

@click.command()
@click.option('--input_filepath', '-i', default=RAW_DATA_PATH, type=str, help='Path to input data.')
@click.option('--output_filepath', '-o', default=PROCESSED_DATA_PATH, type=click.Path(exists=True), help='Path to output data.')
@click.option('--uncertainty_policy', '-u', type=str,
    help='Policy to handle uncertainty.According the CheXpert original paper, policies are "U-Ignore", "U-Zeros", "U-Ones", "U-SelfTrained", and "U-MultiClass".')
@click.option('--hyperparam', '-p', default=None, help='Hyperparameters to train the model in tunning.')
def train(input_filepath: str,
          output_filepath: str,
          uncertainty_policy: str,
          hyperparam: Union[None, Type[Config]]) -> None:
    #TODO: docstring; include model/train params on hyperparam tunning
    logger = logging.getLogger(__name__)

    global config
    if hyperparam is not None:
        config = hyperparam

    os.makedirs(config.ckpt_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)

    num_workers = 1
    pin_memory = False
    seed = 123
    if torch.cuda.is_available():
        logger.info('Cuda available')
        torch.cuda.manual_seed(seed)
        num_workers = 0
        pin_memory = True

    logger.info(f'\nStart training with:\n- Batch size:\t\t{config.batch_size}\n- Uncertainty Policy:\t"{uncertainty_policy}".')
 
    train_dataloader = get_dataloader(data_path=input_filepath,
                                      uncertainty_policy=uncertainty_policy,
                                      logger=logger,
                                      train=True,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory,
                                      resize_shape=config.resize_shape)
    valid_dataloader = get_dataloader(data_path=input_filepath,
                                      uncertainty_policy=uncertainty_policy,
                                      logger=logger,
                                      train=False,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory,
                                      resize_shape=config.resize_shape)
    
    trainer_instance = Trainer(config, (train_dataloader, valid_dataloader))
    trainer_instance.train()

    valid_acc = trainer_instance.best_valid_acc
    return valid_acc


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    train()
