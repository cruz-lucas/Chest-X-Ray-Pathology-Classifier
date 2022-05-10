import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.dataset import CheXpertDataset

RAW_DATA_PATH = 'data/raw/'
PROCESSED_DATA_PATH = 'data/processed/'

@click.command()
@click.option('--input_filepath', '-i', default=RAW_DATA_PATH, type=str, help='Path to input data')
@click.option('--output_filepath', '-o', default=PROCESSED_DATA_PATH, type=click.Path(exists=True), help='Path to output data')
@click.option('--batch_size', '-b', default=128, type=int, help='Batch size in training')
@click.option('--uncertainty_policy', '-u', type=str,
    help='Policy to handle uncertainty.According the CheXpert original paper, policies are "U-Ignore", "U-Zeros", "U-Ones", "U-SelfTrained", and "U-MultiClass"')
def main(input_filepath: str, output_filepath: str, batch_size: int, uncertainty_policy: str) -> None:
    logger = logging.getLogger(__name__)

    logger.info(f'\nStart training with:\n- Batch size:\t\t{batch_size}\n- Uncertainty Policy:\t"{uncertainty_policy}".')
    dataloader = CheXpertDataset(data_path=input_filepath, uncertainty_policy=uncertainty_policy, logger=logger)
    dataloader.__getitem__(0)
    
    return None

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
