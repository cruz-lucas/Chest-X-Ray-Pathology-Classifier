"""Wrapper for CheXpert dataset."""
import io
import logging
from typing import Optional

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from torch.utils.data import Dataset
import torchvision.transforms as T

from google.cloud import storage
from utils import get_config, extract_config
from src import UNCERTAINTY_POLICIES, PATHOLOGIES


class CheXpertDataset(Dataset):
    def __init__(self, config_path: Optional[str] = None, parquet_file_path: Optional[str] = None) -> None:
        """Initialize the dataset and preprocess according to the uncertainty policy.

        Args:
            config_path (Optional[str]): Path to the configuration file. Defaults to None.
            hdf5_file_path (Optional[str]): Path to the HDF5 file. If provided, data will be loaded from this file. Defaults to None.
        """
        self.config = get_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.parquet_file_path = parquet_file_path

        if parquet_file_path:
            self._load_from_parquet(parquet_file_path)
        else:
            self._load_from_raw_data()

    def _load_from_raw_data(self) -> None:
        """Load and preprocess data from raw files."""
        self.uncertainty_policy = extract_config(self.config, 'train', 'uncertainty_policy')
        self.n_labels = len(PATHOLOGIES) if self.uncertainty_policy != 'U-MultiClass' else len(PATHOLOGIES)*3
        self.resize = extract_config(self.config, 'data', 'resize')

        data_path = extract_config(self.config, 'data', 'data_path')
        split = extract_config(self.config, 'data', 'split')
        filename = f"CheXpert-v1.0/{split}.csv"
        path = f"{data_path}/{filename}"
        data = self._load_data(path)
        data.fillna(0, inplace=True)
        # self._preprocess_data(data)

        if 'gs://' in data_path:
            storage_client = storage.Client(project=extract_config(self.config, 'data', 'gcp_project_id'))
            self.bucket = storage_client.bucket(extract_config(self.config, 'data', 'gcp_bucket'))
        else:
            self.bucket = None

        self.image_names = data.index.to_numpy()
        self.labels = data.loc[:, PATHOLOGIES].values.reshape((-1, self.n_labels))

        self.transform = T.Compose([
            T.Resize(self.resize),
            T.ToTensor(),
            T.Normalize(mean=[0.5330], std=[0.0349])
        ])

    def _load_from_parquet(self, parquet_file_path: str) -> None:
        """Load data from a parquet file.

        Args:
            parquet_file_path (str): Path to the parquet file.
        """
        pass

    def _load_data(self, path: str) -> pd.DataFrame:
        """Load data from local or cloud storage.

        Args:
            path (str): Path to the CSV file.
            filename (str): Name of the CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        try:
            data = pd.read_csv(path)
            self.logger.info(f"Database found at {path}")
        except Exception as e:
            self.logger.warning(f"Couldn't read CSV at path {path}.\n{e}")
            raise

        data.set_index('Path', inplace=True)
        return data

    # def _preprocess_data(self, data: pd.DataFrame) -> None:
    #     """Preprocess the data according to the given uncertainty policy.

    #     Args:
    #         data (pd.DataFrame): Data to preprocess.
    #     """
    #     data = data.loc[:, PATHOLOGIES].copy()
    #     data.fillna(0, inplace=True)

    #     if self.uncertainty_policy == 'U-Zeros':
    #         data.replace({-1: 0}, inplace=True)
    #     elif self.uncertainty_policy == 'U-Ones':
    #         data.replace({-1: 1}, inplace=True)
    #     elif self.uncertainty_policy == 'U-MultiClass':
    #         data.loc[:, PATHOLOGIES] = data.applymap(
    #             lambda x: [1., 0., 0.] if x == 0 else [0., 1., 0.] if x == 1 else [0., 0., 1.]
    #         )
    #         data = data.apply(lambda x: np.concatenate(x), axis=1)
    #         self.n_labels = 15

    def __getitem__(self, index: int) -> dict:
        """
        Returns image and label for a given index.

        Args:
            index (int): Index of the sample in the dataset.

        Returns:
            dict: Contains 'pixel_values' (image) and 'labels' (label tensor).
        """
        if self.bucket is None:
            img = Image.open(self.image_names[index]).convert('RGB')
        else:
            img_bytes = self.bucket.blob(self.image_names[index]).download_as_bytes()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')  

        img = self.transform(img)
        label = self.labels[index].astype(np.float32)

        return {"pixel_values": img, "labels": label}

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """

        return len(self.image_names)

    def create_local_database(self, parquet_file_path: str) -> None:
        """Create a local parquet file from the dataset.

        Args:
            parquet_file_path (str): Path to the parquet file to save the dataset.
        """
        for i, batch in enumerate(tqdm(self, total=self.__len__())):
            df_pixels = pd.DataFrame({
                    'pixel_values': batch['pixel_values'].numpy().flatten(),
            })

            df_labels = pd.DataFrame({
                    'labels': batch['labels']
            })
            if i == 0:
                parquet_schema_pixels = pa.Table.from_pandas(df=df_pixels).schema
                parquet_schema_labels = pa.Table.from_pandas(df=df_labels).schema
                parquet_writer_pixels = pq.ParquetWriter(f"{parquet_file_path}_pixels.parquet", parquet_schema_pixels, compression='snappy')
                parquet_writer_labels = pq.ParquetWriter(f"{parquet_file_path}_labels.parquet", parquet_schema_labels, compression='snappy')

            table = pa.Table.from_pandas(df_pixels, schema=parquet_schema_pixels)
            parquet_writer_pixels.write_table(table)

            table = pa.Table.from_pandas(df_labels, schema=parquet_schema_labels)
            parquet_writer_labels.write_table(table)

        parquet_writer_pixels.close()
        parquet_writer_labels.close()
        self.logger.info(f"Data has been successfully saved to {parquet_file_path}_pixels and _labels")
