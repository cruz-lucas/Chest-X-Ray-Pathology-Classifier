from typing import List
import pandas as pd
from PIL import Image
import logging

from torch import FloatTensor
from torch.utils.data import Dataset
from torch.cuda import is_available

use_gpu = is_available()

# CheXpert pathologies on paper
pathologies = ['atelectasis',
               'cardiomegaly',
               'consolidation',
               'edema',
               'pleural effusion']


######################
## Create a Dataset ##
######################
class CheXpertDataset(Dataset):
    def __init__(self, data_path: str, uncertainty_policy: str, logger: logging.Logger, pathologies: List[str] = pathologies) -> None:
        """ Innitialize dataset and preprocess according to uncertainty policy.

        Args:
            data_path (str): Path to csv file.
            uncertainty_policy (str): Uncertainty policies compared in the original paper.
            Check if options are implemented. Options: 'U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', and 'U-MultiClass'.
            logger (logging.Logger): Logger to log events during training.
            pathologies (List[str], optional): Pathologies to classify.
            Defaults to 'atelectasis', 'cardiomegaly', 'consolidation', 'edema', and 'pleural effusion'.

        Returns:
            None
        """

        uncertainty_policies = ['U-Ignore',
                                'U-Zeros',
                                'U-Ones',
                                'U-SelfTrained',
                                'U-MultiClass']
        
        if not(uncertainty_policy in uncertainty_policies):
            logger.error(f"Unknown uncertainty policy. Known policies: {uncertainty_policies}")
            return None

        try:
            data = pd.read_csv(data_path)
        except Exception as e:
            logger.error(f"Couldn't read csv at path {data_path}.\n{e}")
            return None

        data.set_index('path', inplace=True)
        data = data.loc[:, pathologies].copy()

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

    def __getitem__(self, index: int):
        """ Returns image and label from given index.

        Args:
            index (int): Index of sample in dataset.

        Returns:
            _type_: _description_
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        return image, FloatTensor(label)

    def __len__(self) -> int:
        """ Return length of dataset.

        Returns:
            int: length of dataset.
        """
        return len(self.image_names)