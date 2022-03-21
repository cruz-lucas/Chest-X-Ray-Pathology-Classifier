import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from datetime import date

import pandas as pd
import numpy as np

from src.data.dataset import Dataset

from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split

import optuna
import ast

RAW_DATA_PATH = 'data/raw/'
PROCESSED_DATA_PATH = 'data/processed/'

def online_ocsvm(input_filepath: str, data: pd.DataFrame, target_class: str, batch_size: int, resize_dimension: tuple, logger:logging.Logger):

    def partial_pipe_fit(pipeline_obj, df):
        kernel_mapper = pipeline_obj.named_steps['kernel_mapper'].fit_transform(df)
        pipeline_obj.named_steps['clf'].partial_fit(kernel_mapper)

        return pipeline_obj

    def objective(trial):
        # TODO: test normalization (batch norm?)
        
        # Hyperparameters
        kernel = 'rbf'
        gamma = 0.1
        random_state = 123
        nu = 0.1
        learning_rate = 'optimal' # constant, optimal, invscaling, adaptive

        transform = Nystroem(kernel=kernel, gamma=gamma, random_state=random_state) 
        clf_sgd = SGDOneClassSVM(
            nu=nu, shuffle=True, fit_intercept=True, random_state=random_state, tol=1e-4,
            learning_rate=learning_rate,
        )
        pipe_sgd = Pipeline([('kernel_mapper', transform), ('clf', clf_sgd)])
        
        number_of_batches = len(data) // batch_size
        for batch_index in range(number_of_batches):
            logger.info(f'Batch {batch_index} of {number_of_batches}')
            batch = data.iloc[batch_index*batch_size:(batch_index+1)*batch_size].copy()
            processed = Dataset(batch, input_filepath, resize_dimension).processed

            # TODO: Check partial fit on pipeline
            #pipe_sgd = partial_pipe_fit(pipe_sgd, X_train)

    study_name = 'test'#date.today().strftime("%Y_%m_%d_%H_%M")
    study = optuna.create_study(direction='maximize', study_name=f'{study_name}__hyperparameters_online_ocsvm', load_if_exists=True)
    study.optimize(objective, n_trials=100)


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.command()
@click.option('--input_filepath', '-i', default=RAW_DATA_PATH, type=click.Path(exists=True), help='Path to input data')
@click.option('--output_filepath', '-o', default=PROCESSED_DATA_PATH, type=click.Path(exists=True), help='Path to output data')
@click.option('--resize_dimension', '-r', cls=PythonLiteralOption, default=(128,128), type=tuple, help='Dimension to resize images')
@click.option('--batch_size', '-b', default=128, type=int, help='Batch size in training')
@click.option('--target', '-t', default='Cardiomegaly', type=str, help='Target class (one class) to train one versus all')
@click.option('--uncertainty', '-u', type=click.Choice(['ignore', 'zeros', 'ones', 'selftrained', 'multiclass'], case_sensitive=False), help='Target class (one class) to train one versus all')
def main(input_filepath: str, output_filepath: str, resize_dimension: tuple, batch_size: int, target: str, uncertainty: str) -> None:
    logger = logging.getLogger(__name__)

    # Train dataset
    logger.info(f'Start training with resize dimensions {resize_dimension}, batch size {batch_size}, for target {target} and handling uncertainty with {uncertainty} approach.')

    data = pd.read_csv(input_filepath+'/CheXpert-v1.0/train.csv')

    if uncertainty == 'multiclass':
        raise NotImplementedError('Not Implemented multiclass for handling uncertainty.')

    elif uncertainty == 'selftrained':
        raise NotImplementedError('Not Implemented selftraining for handling uncertainty.')

    elif uncertainty == 'ignore':
        data = data[(data[target] != -1)].reset_index(drop=True)
        data[target] = np.where(data[target] == 1, 1, -1)

    elif uncertainty == 'zeros':
        data[target] = np.where(data[target] == 1, 1, -1)

    elif uncertainty == 'ones':
        data[target] = np.where((data[target] == 1) & (data[target] == -1), 1, 0)
        data[target] = np.where(data[target] == 1, 1, -1)
    
    try:
        train = pd.read_parquet(f'{output_filepath}{target}_{uncertainty}_train.parquet')
        holdout = pd.read_parquet(f'{output_filepath}{target}_{uncertainty}_holdout.parquet')
        logger.info(f'Train and holdout sets loaded.')
    except FileNotFoundError:
        logger.info(f'Train and holdout sets not found, splitting sets...')
        train, holdout = train_test_split(data, test_size=0.2, random_state=123, stratify=data[target])
        train.to_parquet(f'{output_filepath}{target}_{uncertainty}_train.parquet')
        holdout.to_parquet(f'{output_filepath}{target}_{uncertainty}_holdout.parquet')
    #online_ocsvm(input_filepath, data, target, batch_size, resize_dimension, logger)

    # Validation dataset
    #logger.info('making final data set for validation from raw data')

    #data = pd.read_csv(input_filepath+'/CheXpert-v1.0/valid.csv')
    #run_preprocess(data, input_filepath, output_filepath, resize_dimension)

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
