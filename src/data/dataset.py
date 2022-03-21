import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
ProgressBar().register()

from src.features.build_features import read_image

NPARTITIONS = 30

class Dataset(object):
    def __init__(self, data: pd.DataFrame, input_filepath: str, resize_dimension: tuple):
        data.Path = input_filepath + '/' + data.Path
        data['resize_dimension'] = [(resize_dimension[0],resize_dimension[-1])] * len(data)
        self.data = data
        self.input_filepath = input_filepath
        self.resize_dimension = resize_dimension
        self.processed = None
        self.run_preprocess()

    def run_preprocess(self) -> None:
        ## test and develop pipeline, data is filter to only 100 images
        processed = self.data[self.data['Frontal/Lateral'] == 'Frontal'].copy()

        ddata = dd.from_pandas(processed, npartitions=NPARTITIONS)
        processed['img'] = ddata.map_partitions(lambda df: df.apply(read_image, axis=1)).compute(scheduler='threads')

        self.processed = processed
