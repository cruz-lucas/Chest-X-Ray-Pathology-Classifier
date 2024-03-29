import io
from typing import List, Union

import pandas as pd
import numpy as np
from PIL import Image

from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T

from google.cloud import storage

import logging

# CheXpert pathologies on original paper
pathologies = ['Atelectasis',
               'Cardiomegaly',
               'Consolidation',
               'Edema',
               'Pleural Effusion']

# Uncertainty policies on original paper
uncertainty_policies = ['U-Ignore',
                        'U-Zeros',
                        'U-Ones',
                        'U-SelfTrained',
                        'U-MultiClass']


# #####################
# # Create a Dataset ##
# #####################
class CheXpertDataset(Dataset):
    def __init__(self,
                 data_path: Union[str, None] = None,
                 uncertainty_policy: str = 'U-Ones',
                 logger: logging.Logger = logging.getLogger(__name__),
                 pathologies: List[str] = pathologies,
                 train: bool = True,
                 csv_name: str = None,
                 resize_shape: tuple = (256, 256)) -> None:
        """ Innitialize dataset and preprocess according to uncertainty policy.

        Args:
            data_path (str): Path to csv file.
            uncertainty_policy (str): Uncertainty policies compared in the
            original paper.
            Check if options are implemented. Options: 'U-Ignore', 'U-Zeros',
            'U-Ones', 'U-SelfTrained', and 'U-MultiClass'.
            logger (logging.Logger): Logger to log events during training.
            pathologies (List[str], optional): Pathologies to classify.
            Defaults to 'Atelectasis', 'Cardiomegaly', 'Consolidation',
            'Edema', and 'Pleural Effusion'.
            transform (type): method to transform image.
            train (bool): If true, returns data selected for training, if not,
            returns data selected for validation (dev set), as the CheXpert
            research group splitted.

        Returns:
            None
        """

        if not (uncertainty_policy in uncertainty_policies):
            logger.error(
                "Unknown uncertainty policy. Known policies: " +
                f"{uncertainty_policies}")
            return None

        split = csv_name if csv_name is not None else 'train' if train else 'valid'
        csv_path = f"CheXpert-v1.0/{split}.csv"
        path = str(data_path) + csv_path

        self.in_cloud = False

        data = pd.DataFrame()
        try:
            data = pd.read_csv(path)
            if csv_name == 'test_labels':
                data['Path'] = data_path + 'CheXpert-v1.0/' + data['Path']
            else:
                data['Path'] = data_path + data['Path']
            logger.info(f"Local database found at {path}")
        except Exception as e:
            logger.warning(f"Couldn't read csv at path {path}./n{e}")
            try:
                # Find files at gcp
                project_id = 'labshurb'

                storage_client = storage.Client(project=project_id)
                self.bucket = storage_client.bucket(
                    'chexpert_database_stanford')

                blob = self.bucket.get_blob(csv_path)
                blob.download_to_filename('tmp.csv')
                data = pd.read_csv('tmp.csv')

                self.in_cloud = True
                logger.info("Cloud database found.")

            except Exception as e_:
                logger.error(f"Couldn't reach file at path {path}./n{e_}")
                quit()

        data.set_index('Path', inplace=True)

        # data = data.loc[data['Frontal/Lateral'] == 'Frontal'].copy()
        data = data.loc[:, pathologies].copy()

        # it will change for 15 in case of multiclass
        label_cols = 5

        data.fillna(0, inplace=True)

        # U-Ignore
        if uncertainty_policy == uncertainty_policies[0]:
            # the only change is in the loss function, we mask the -1 labels
            # in the calculation
            pass

        # U-Zeros
        elif uncertainty_policy == uncertainty_policies[1]:
            data.replace({-1: 0}, inplace=True)

        # U-Ones
        elif uncertainty_policy == uncertainty_policies[2]:
            data.replace({-1: 1}, inplace=True)

        # U-SelfTrained
        elif uncertainty_policy == uncertainty_policies[3]:
            # No action needed
            pass

        # U-MultiClass
        elif uncertainty_policy == uncertainty_policies[4]:
            #data.replace({-1: 2}, inplace=True)

            one_hot_0 = [1., 0., 0.]
            one_hot_1 = [0., 1., 0.]
            one_hot_2 = [0., 0., 1.]

            data.loc[:, pathologies] = data.map(lambda x: one_hot_0 if x == 0 else one_hot_1 if x == 1 else one_hot_2).to_numpy()

            label_cols = 15

        self.image_names = data.index.to_numpy()
        self.labels = np.array(
            data.loc[:, pathologies].values.tolist()
            ).reshape((-1, label_cols))
        self.transform = T.Compose([
                  T.Resize(resize_shape),
                  T.ToTensor(),
                  T.Normalize(mean=[0.5330], std=[0.0349])
              ])  # whiten with dataset mean and stdif transform)


    def __getitem__(self, index: int) -> Union[np.array, Tensor]:
        """ Returns image and label from given index.

        Args:
            index (int): Index of sample in dataset.

        Returns:
            np.array: Array of grayscale image.
            torch.Tensor: Tensor of labels.
        """
        if self.in_cloud:
            img_bytes = self.bucket.blob(
                self.image_names[index]).download_as_bytes()
            # .download_to_filename('tmp.jpg')
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        else:
            img = Image.open(self.image_names[index]).convert('RGB')
        img = self.transform(img)

        label = self.labels[index].astype(np.float32)
        return {"pixel_values": img, "labels": label}

    def __len__(self) -> int:
        """ Return length of dataset.

        Returns:
            int: length of dataset.
        """
        return len(self.image_names)