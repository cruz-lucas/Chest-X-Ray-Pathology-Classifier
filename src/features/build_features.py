import cv2
import numpy as np
import pandas as pd

import os
from typing import Union

def read_image(row) -> Union[np.ndarray, None]:
    """ Read image and if a valid path is provided (just contains '.jpg' for now, 
    more error handling will be implemented) returns an image in a 2d np.ndarray.

    Args:
        any tuple or list with 1 element (any position) that is a valid path to an
        image.

    Returns:
        Union[np.ndarray, None]: return image or None if not a valid path
    """
    reduced_size = (row.resize_dimension[0],row.resize_dimension[-1])
    
    path = row.Path
    
    if (path is not None) and (row.Path != 'foo'):
        # list not empty nor is type inference for dask's map partition
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, reduced_size, interpolation=cv2.INTER_LINEAR)
        img = img.reshape(-1, )

        return img   
    return None
