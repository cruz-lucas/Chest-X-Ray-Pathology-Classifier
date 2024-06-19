"""Wrapper for CheXpert dataset."""
import io
import logging
from typing import Optional

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import webdataset as wds
import sys

from torch import from_numpy
from torch.utils.data import Dataset
import torchvision.transforms as T

from google.cloud import storage
from utils import get_config, extract_config
from src import UNCERTAINTY_POLICIES, PATHOLOGIES


class CheXpertDataset(Dataset):
    def __init__(self, config_path: Optional[str] = None, webdataset_file_path: Optional[str] = None) -> None:
        """Initialize the dataset and preprocess according to the uncertainty policy.

        Args:
            config_path (Optional[str]): Path to the configuration file. Defaults to None.
            hdf5_file_path (Optional[str]): Path to the HDF5 file. If provided, data will be loaded from this file. Defaults to None.
        """
        self.config = get_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.webdataset_file_path = webdataset_file_path

        if webdataset_file_path:
            self._load_from_webdataset(webdataset_file_path)
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

    def _load_from_webdataset(self) -> None:
        """Load data from a parquet file.
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

    def create_local_database(self, webdataset_file_path: str) -> None:
        """Create a local parquet file from the dataset.

        Args:
            webdataset_file_path (str): Path to the parquet file to save the dataset.
        """
        sink = wds.TarWriter(f"{webdataset_file_path}.tar")
        for index in tqdm(range(self.__len__())):
            if index%1000==0:
                print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)

            sample = self.__getitem__(index)

            sink.write({
                "__key__": "sample%06d" % index,
                "img.pth": sample['pixel_values'],
                "labels.pth": from_numpy(sample['labels']),
            })

            if index == 100:
                break

        sink.close()
        self.logger.info(f"Data has been successfully saved to {webdataset_file_path}_pixels and _labels")
