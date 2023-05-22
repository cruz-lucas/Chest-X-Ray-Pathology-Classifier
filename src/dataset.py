import logging
import io
from typing import List, Union
import pandas as pd
import numpy as np
from PIL import Image

from torch import from_numpy, Tensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as T

from google.cloud import storage

import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


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

######################
## Create a Dataset ##
######################
class CheXpertDataset(Dataset):
    def __init__(self,
                 data_path: Union[str, None] = None,
                 uncertainty_policy: str = 'U-Ones',
                 logger: logging.Logger = logging.getLogger(__name__),
                 pathologies: List[str] = pathologies,
                 train: bool = True,
                 resize_shape: tuple = (384, 384)) -> None:
        """ Innitialize dataset and preprocess according to uncertainty policy.

        Args:
            data_path (str): Path to csv file.
            uncertainty_policy (str): Uncertainty policies compared in the original paper.
            Check if options are implemented. Options: 'U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', and 'U-MultiClass'.
            logger (logging.Logger): Logger to log events during training.
            pathologies (List[str], optional): Pathologies to classify.
            Defaults to 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', and 'Pleural Effusion'.
            transform (type): method to transform image.
            train (bool): If true, returns data selected for training, if not, returns data selected for validation (dev set), as the CheXpert research group splitted.

        Returns:
            None
        """
        
        if not(uncertainty_policy in uncertainty_policies):
            logger.error(f"Unknown uncertainty policy. Known policies: {uncertainty_policies}")
            return None
        
        project_id = 'labshurb'

        storage_client = storage.Client(project=project_id)
        self.bucket = storage_client.bucket('chexpert_database_stanford')

        split = 'train' if train  else 'valid'
        csv_path = f"CheXpert-v1.0/{split}.csv"
        path = str(data_path) + csv_path

        data = pd.DataFrame()
        try:
            data = pd.read_csv(path)
        except Exception as e:
            try:
              blob = self.bucket.get_blob(csv_path)
              blob.download_to_filename('tmp.csv')
              data = pd.read_csv('tmp.csv')
            except:  
              logger.error(f"Couldn't read csv at path {path}.\n{e}")
              quit()

        data['Path'] = data['Path'] # data_path + 
        data.set_index('Path', inplace=True)

        #data = data.loc[data['Frontal/Lateral'] == 'Frontal'].copy()
        data = data.loc[:, pathologies].copy()
        
        data.fillna(0, inplace=True)

        # U-Ignore
        if uncertainty_policy == uncertainty_policies[0]:
            data = data.loc[(data[pathologies] != -1).all(axis=1)].copy()
        
        # U-Zeros
        elif uncertainty_policy == uncertainty_policies[1]:
            data.replace({-1: 0}, inplace=True)

        # U-Ones
        elif uncertainty_policy == uncertainty_policies[2]:
            data.replace({-1: 1}, inplace=True)

        # U-SelfTrained
        elif uncertainty_policy == uncertainty_policies[3]:
            logger.error(f"Uncertainty policy {uncertainty_policy} not implemented.")
            return None

        # U-MultiClass
        elif uncertainty_policy == uncertainty_policies[4]:
            # Do nothing and leave -1 as a label, but check if whole system works.
            logger.error(f"Uncertainty policy {uncertainty_policy} not implemented.")
            return None

        self.image_names = data.index.to_numpy()
        self.labels = data.loc[:, pathologies].to_numpy()
        self.transform = T.Compose([
                  T.Resize(resize_shape),
                  T.ToTensor(),
                  T.Normalize(mean=[0.5330], std=[0.0349])
              ]) # whiten with dataset mean and stdif transform)

    def __getitem__(self, index: int) -> Union[np.array, Tensor]:
        """ Returns image and label from given index.

        Args:
            index (int): Index of sample in dataset.

        Returns:
            np.array: Array of grayscale image.
            torch.Tensor: Tensor of labels.
        """
        img_bytes = self.bucket.blob(self.image_names[index]).download_as_bytes()#.download_to_filename('tmp.jpg')
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = self.transform(img)

        label = from_numpy(self.labels[index].astype(np.float32))
        return img, label

    def __len__(self) -> int:
        """ Return length of dataset.

        Returns:
            int: length of dataset.
        """
        return len(self.image_names)


#########################
## Create a DataLoader ##
#########################
def get_dataloader(data_path: str,
                   uncertainty_policy: str,
                   batch_size: int,
                   pathologies: List[str] = pathologies,
                   train: bool = True,
                   shuffle: bool = True,
                   random_seed: int = 123,
                   num_workers: int = 4, 
                   pin_memory: bool = True,
                   resize_shape: tuple = (384, 384)):
    """Get wrap dataset with dataloader class to help with paralellization, data loading order 
    (for reproducibility) and makes the code o bit cleaner.

    Args:
        data_path (str): Refer to CheXpertDataset class documentation.
        uncertainty_policy (str): Refer to CheXpertDataset class documentation.
        logger (logging.Logger): Refer to CheXpertDataset class documentation.
        pathologies (List[str], optional): Refer to CheXpertDataset class documentation.
        train (bool): Refer to CheXpertDataset class documentation.
        shuffle (bool): Shuffle datasets (independently, train or valid).
        random_seed (int): Seed to shuffle data, helps with reproducibility.

    Returns:
        torch.utils.data.DataLoader: Data loader from dataset randomly (or not) loaded.
    """

    dataset = CheXpertDataset(
        data_path=data_path,
        uncertainty_policy=uncertainty_policy,
        pathologies=pathologies,
        train=train,
        resize_shape=resize_shape
        )
    
    indices = list(range(dataset.__len__()))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    sampler = SubsetRandomSampler(indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        )
