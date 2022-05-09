from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

import optuna

from src.data.dataset import Dataset

import ast
import pandas as pd
import numpy as np
import logging

from math import ceil

class Online_OCSVM():
    def __init__(self, input_filepath: str, data: pd.DataFrame, target_class: str, batch_size: int, resize_dimension: tuple, logger:logging.Logger):
        self.input_filepath = input_filepath
        self.data = data
        self.target_class = target_class
        self.batch_size = batch_size
        self.resize_dimension = resize_dimension
        self.logger = logger

    def partial_pipe_fit(self, pipeline_obj, df):
        kernel_mapper = pipeline_obj.named_steps['kernel_mapper'].fit_transform(df)
        pipeline_obj.named_steps['clf'].partial_fit(kernel_mapper)

        return pipeline_obj

    def fit_ocsvm(self, kernel, gamma, random_state, nu, learning_rate):
        transform = Nystroem(kernel=kernel, gamma=gamma, random_state=random_state) 
        clf_sgd = SGDOneClassSVM(
            nu=nu, shuffle=True, fit_intercept=True, random_state=random_state, tol=1e-4,
            learning_rate=learning_rate,
        )
        pipe_sgd = Pipeline([('kernel_mapper', transform), ('clf', clf_sgd)])
        
        target_class_data = self.data[self.data[self.target_class] == 1].copy()
        number_of_batches = ceil(len(target_class_data) / self.batch_size)
        for batch_index in range(number_of_batches):
            self.logger.info(f'Batch {batch_index} of {number_of_batches}')

            batch = target_class_data.iloc[batch_index*self.batch_size:(batch_index+1)*self.batch_size].copy()
            X_train = Dataset(batch, self.input_filepath, self.resize_dimension).processed.img.to_numpy()

            pipe_sgd = self.partial_pipe_fit(pipe_sgd, X_train)
            print(pipe_sgd)

        # TODO: calculate score
        return 0 #score

    def objective(self, trial):
        # TODO: test normalization (batch norm?)
        
        # Hyperparameters
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
        gamma = trial.suggest_float("gamma", low=0.001, high=0.5, log=False)
        random_state = 123
        nu = trial.suggest_float("nu", low=0.001, high=0.999, log=False)
        learning_rate = trial.suggest_categorical("learning_rate", ["constant", "optimal", "invscaling", "adaptive"])

        # TODO: set cross validation 
        score = self.fit_ocsvm(kernel, gamma, random_state, nu, learning_rate)

        return score
        
    def search_hyperparameters(self, study_name):
        study = optuna.create_study(direction='maximize', study_name=f'{study_name}__hyperparameters_online_ocsvm', load_if_exists=True)
        study.optimize(self.objective, n_trials=100)