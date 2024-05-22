from chexpert import CheXpertDataset
import h5py
import numpy as np
from tqdm import tqdm


def create_local_database() -> None:
    dataloader = CheXpertDataset()

    with h5py.File(f'dataset_{dataloader.uncertainty_policy}.h5', 'w') as hdf5_file:
        # Create datasets in the HDF5 file
        pixel_values_dataset = hdf5_file.create_dataset(
            'pixel_values',
            shape=(dataloader.__len__(), 3, dataloader.resize[0], dataloader.resize[1]),
            dtype=np.float32
        )
        label_dataset = hdf5_file.create_dataset(
            'labels',
            shape=(dataloader.__len__(), dataloader.n_labels),
            dtype=np.int64
        )

        # Loop through the DataLoader and write data to the HDF5 file
        for i, batch in enumerate(tqdm(dataloader, total=dataloader.__len__())):
            pixel_values = batch['pixel_values'].numpy()
            labels = batch['labels']

            pixel_values_dataset[i] = pixel_values
            label_dataset[i] = labels
