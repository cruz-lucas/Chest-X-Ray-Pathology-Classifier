import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.dataset import get_dataloader

import torch

kwargs = {}
seed = 123
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    kwargs = {"num_workers": 1, "pin_memory": True}


# Data paths default values
RAW_DATA_PATH = 'data/raw/'
PROCESSED_DATA_PATH = 'data/processed/'

@click.command()
@click.option('--input_filepath', '-i', default=RAW_DATA_PATH, type=str, help='Path to input data.')
@click.option('--output_filepath', '-o', default=PROCESSED_DATA_PATH, type=click.Path(exists=True), help='Path to output data.')
@click.option('--uncertainty_policy', '-u', type=str,
    help='Policy to handle uncertainty.According the CheXpert original paper, policies are "U-Ignore", "U-Zeros", "U-Ones", "U-SelfTrained", and "U-MultiClass".')
@click.option('--batch_size', '-b', default=128, type=int, help='Batch size in training.')
@click.option('--img_size', '-s', default=128, type=int, help='Image size.')
@click.option('--resume', '-r', default=False, type=bool, help='Flag to resume previous started training.')
@click.option('--checkpoint', '-cp', default=None, type=str, help='Checkpoint to resume training.')
@click.option('--learning_rate', '-lr', default=1e-3, type=float, help='Learning rate for training.')
@click.option('--epochs', '-e', default=200, type=int, help='Epochs for training.')
@click.option('--shuffle', '-sh', default=True, type=bool, help='Shuffle train and valid datasets (independently).')
@click.option('--random_seed', '-rs', default=seed, type=int, help='Seed to shuffle data, helps with reproducibility.')
def train(input_filepath: str,
          output_filepath: str,
          uncertainty_policy: str,
          batch_size: int,
          img_size: int,
          resume: bool,
          checkpoint: str,
          learning_rate: float,
          epochs: int,
          shuffle: bool,
          ) -> None:
    #TODO: docstring; include model/train params on hyperparam tunning
    logger = logging.getLogger(__name__)

    logger.info(f'\nStart training with:\n- Batch size:\t\t{batch_size}\n- Uncertainty Policy:\t"{uncertainty_policy}".')
 
    train_dataloader = get_dataloader(data_path=input_filepath,
                                      uncertainty_policy=uncertainty_policy,
                                      logger=logger,
                                      train=True,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      random_seed=random_seed,
                                      **kwargs=**kwargs)
    valid_dataloader = get_dataloader(data_path=input_filepath,
                                      uncertainty_policy=uncertainty_policy,
                                      logger=logger,
                                      train=False,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      random_seed=random_seed,
                                      **kwargs=**kwargs)

    config = {
                'use_gpu': True,
                'patch_size': ,
                'glimpse_scale': ,
                'num_patches': ,
                'loc_hidden': ,
                'glimpse_hidden': ,
                'num_glimpses': ,
                'hidden_size': ,
                'std': ,
                'M': ,
                'is_train': True,
                'epochs': ,
                'momentum': ,
                'init_lr': ,
                'best': ,
                'ckpt_dir': ,
                'logs_dir': ,
                'lr_patience': ,
                'train_patience': ,
                'use_tensorboard': ,
                'resume': ,
                'print_freq': ,
                'plot_freq': ,
             }
    
    trainer_instance = trainer(config, (train_dataloader, valid_dataloader))
    trainer_instance.num_classes = 5
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
