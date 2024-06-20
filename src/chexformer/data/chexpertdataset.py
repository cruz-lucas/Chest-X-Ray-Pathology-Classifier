"""Wrapper for CheXpert dataset."""
import io
import logging
import sys

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from google.cloud import storage
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from chexformer.utils import PreprocessConfig


class CheXpertDataset(Dataset):
    """Class to aggregate data handling methods.

    Args:
        Dataset (Dataset): Pytorch's dataset.
    """

    def __init__(self, config: PreprocessConfig) -> None:
        """Initialize the dataset and preprocess according to the uncertainty policy.

        Args:
            config (PreprocessConfig): Configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_from_raw_data(self) -> None:
        """Load and preprocess data from raw files."""
        path = f"{self.config.data_path}/CheXpert-v1.0/{self.config.split}.csv"
        data = self._load_data(path)
        data.fillna(0, inplace=True)

        if "gs://" in self.config.data_path:
            storage_client = storage.Client(project=self.config.gcp_project_id)
            self.bucket = storage_client.bucket(self.config.gcp_bucket)
        else:
            self.bucket = None

        self.image_names = data.index.to_numpy()
        self.labels = data.loc[:, self.config.constants.pathologies].values.reshape(
            (-1, len(self.config.constants.pathologies))
        )

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(tuple(self.config.resize)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5330], std=[0.0349]),
            ]
        )

    def load_from_webdataset(self) -> wds.WebDataset:
        """Load data from a tar file.

        Returns:
            wds.WebDataset: dataset loaded from tar file.
        """

        def unpack(data):
            image = data["img.pth"]
            labels = data["labels.pth"]
            return image, labels

        return wds.WebDataset(self.config.data_path).decode("torch").map(unpack)

    def preprocess_dataset(self) -> None:
        """Create a local preprocessed dataset from the original dataset."""
        sink = wds.TarWriter(f"{self.config.dataset_dir}/chexpert_{self.config.split}.tar")
        for index in tqdm(range(self.__len__())):
            if index % 1000 == 0:
                print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)

            sample = self.__getitem__(index)

            sink.write(
                {
                    "__key__": "sample%06d" % index,
                    "img.pth": sample["pixel_values"],
                    "labels.pth": torch.from_numpy(sample["labels"]),
                }
            )

        sink.close()
        self.logger.info(
            f"Data has been successfully saved to {self.config.dataset_dir}/chexpert_{self.config.split}.tar"
        )

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

        data.set_index("Path", inplace=True)
        return data

    def __getitem__(self, index: int) -> dict:
        """Returns image and label for a given index.

        Args:
            index (int): Index of the sample in the dataset.

        Returns:
            dict: Contains 'pixel_values' (image) and 'labels' (label tensor).
        """
        if self.bucket is None:
            img = Image.open(self.image_names[index]).convert("RGB")
        else:
            img_bytes = self.bucket.blob(self.image_names[index]).download_as_bytes()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img = self.transform(img)
        label = self.labels[index].astype(np.float32)

        return {"pixel_values": img, "labels": label}

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.image_names)
