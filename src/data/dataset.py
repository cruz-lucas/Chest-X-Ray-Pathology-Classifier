import logging
from pathlib import PurePath
from typing import List, Union
import pandas as pd
import numpy as np
import cv2

from torch import FloatTensor, Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

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
                 data_path: str,
                 uncertainty_policy: str,
                 logger: logging.Logger,
                 pathologies: List[str] = pathologies,
                 transform = None,
                 train: bool = True) -> None:
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

        split = 'train' if train  else 'valid'
        path = PurePath(data_path, 'CheXpert-v1.0/{split}.csv')
        try:
            data = pd.read_csv(path)
        except Exception as e:
            logger.error(f"Couldn't read csv at path {path}.\n{e}")
            return None

        data['Path'] = data_path + data['Path']
        data.set_index('Path', inplace=True)
        data = data.loc[:, pathologies].copy()
        data.fillna(0, inplace=True)

        # U-Ignore
        if uncertainty_policy == uncertainty_policies[0]:
            logger.error(f"Uncertainty policy {uncertainty_policy} not implemented.")
            return None
        
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
            pass # Do nothing and leave -1 as a label

        self.image_names = data.index.to_numpy()
        self.labels = data.loc[:, pathologies].to_numpy()
        self.transform = transform
        self.__len = len(self.image_names)

    def __getitem__(self, index: int) -> Union[np.array, Tensor]:
        """ Returns image and label from given index.

        Args:
            index (int): Index of sample in dataset.

        Returns:
            np.array: Array of grayscale image.
            torch.Tensor: Tensor of labels.
        """
        image = cv2.imread(self.image_names[index], 0)
        if self.transform:
            image = self.transform(image)

        # Norm between -1.0 and 1.0
        image = (np.array(image) - 128.0)/128.0
        label = self.labels[index]
        return image, FloatTensor(label)

    def __len__(self) -> int:
        """ Return length of dataset.

        Returns:
            int: length of dataset.
        """
        return self.__len


#########################
## Create a DataLoader ##
#########################
def get_dataloader(self,
                   data_path: str,
                   uncertainty_policy: str,
                   logger: logging.Logger,
                   batch_size: int,
                   pathologies: List[str] = pathologies,
                   transform = None,
                   train: bool = True,
                   shuffle: bool = True,
                   random_seed: int = 123,
                   num_workers: int = 4, 
                   pin_memory: bool = False,
                   apply_transform: bool = True):
    """Get wrap dataset with dataloader class to help with paralellization, data loading order 
    (for reproducibility) and makes the code o bit cleaner.

    Args:
        data_path (str): Refer to CheXpertDataset class documentation.
        uncertainty_policy (str): Refer to CheXpertDataset class documentation.
        logger (logging.Logger): Refer to CheXpertDataset class documentation.
        pathologies (List[str], optional): Refer to CheXpertDataset class documentation.
        transform (type): Refer to CheXpertDataset class documentation.
        train (bool): Refer to CheXpertDataset class documentation.
        shuffle (bool): Shuffle datasets (independently, train or valid).
        random_seed (int): Seed to shuffle data, helps with reproducibility.

    Returns:
        torch.utils.data.DataLoader: Data loader from dataset randomly (or not) loaded.
    """
    transform = T.compose([T.lambda(lambda x: x)])
    if apply_transform:
        transform = T.compose([lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255).unsqueeze(0),   # tensor in [0,1]
                            T.Normalize(mean=[0.5330], std=[0.0349])])                                             # whiten with dataset mean and stdif transform)

    dataset = CheXpertDataset(data_path=input_filepath, uncertainty_policy=uncertainty_policy, logger=logger, train=train)
    
    indices = list(range(dataset.__len__()))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    sampler = SubsetRandomSampler(indices)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      sampler=sampler,
                      num_workers=num_workers,
                      pin_memory=pin_memory,)