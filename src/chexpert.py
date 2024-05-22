"""Wrapper for CheXpert dataset."""
import io
import logging

import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as T

from google.cloud import storage
from utils import get_config, extract_config
from src import UNCERTAINTY_POLICIES, PATHOLOGIES


class CheXpertDataset(Dataset):
    def __init__(self) -> None:
        """ Initialize dataset and preprocess according to uncertainty policy.
        """
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.uncertainty_policy = extract_config(self.config, 'train', 'uncertainty_policy')
        self.n_labels = len(PATHOLOGIES)
        self.resize = extract_config(self.config, 'data', 'resize')

        data_path = extract_config(self.config, 'data', 'data_path')
        split = extract_config(self.config, 'data', 'split')
        filename = f"CheXpert-v1.0/{split}.csv"
        path = f"{data_path}/{filename}"
        data = self._load_data(path, filename)
        self._preprocess_data(data)

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

    def _validate_config(self):
        if self.uncertainty_policy not in UNCERTAINTY_POLICIES:
            self.logger.error(
                f"Unknown uncertainty policy. Known policies: {UNCERTAINTY_POLICIES}"
            )
            raise ValueError(f"Unknown uncertainty policy: {self.uncertainty_policy}")

    def _load_data(self, path, filename):
        """Load data from local or cloud storage."""
        try:
            data = pd.read_csv(path)
            self.logger.info(f"Database found at {path}")
        except Exception as e:
            self.logger.warning(f"Couldn't read CSV at path {path}.\n{e}")
            raise

        data.set_index('Path', inplace=True)
        return data

    def _preprocess_data(self, data):
        """Preprocess the data according to the given uncertainty policy."""
        data = data.loc[:, PATHOLOGIES].copy()
        data.fillna(0, inplace=True)

        if self.uncertainty_policy == 'U-Zeros':
            data.replace({-1: 0}, inplace=True)
        elif self.uncertainty_policy == 'U-Ones':
            data.replace({-1: 1}, inplace=True)
        elif self.uncertainty_policy == 'U-MultiClass':
            data.loc[:, PATHOLOGIES] = data.applymap(
                lambda x: [1., 0., 0.] if x == 0 else [0., 1., 0.] if x == 1 else [0., 0., 1.]
            )
            data = data.apply(lambda x: np.concatenate(x), axis=1)
            self.n_labels = 15

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
        """Returns the length of the dataset."""
        return len(self.image_names)
