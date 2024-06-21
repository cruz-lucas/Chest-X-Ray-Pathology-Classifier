"""Data preprocessing workflow."""
import json

from chexformer.data import CheXpertDataset
from chexformer.utils import Constants, PreprocessConfig


def preprocess_dataset_workflow(config: PreprocessConfig, constants: Constants):
    """Run preprocess workflow and persists dataset locally.

    Args:
        config (PreprocessConfig): Configuration for preprocessing data.
        constants (Constants): Constants used in the project.
    """
    dataset = CheXpertDataset(config=config, constants=constants)
    dataset.load_from_raw_data()
    dataset.preprocess_dataset()


if __name__ == "__main__":
    with open("./src/chexformer/config/constants.json") as f:
        const_json_obj = json.load(f)
    with open("./src/chexformer/config/preprocess_config.json") as f:
        json_obj = json.load(f)
    preprocess_dataset_workflow(config=PreprocessConfig(**json_obj), constants=Constants(**const_json_obj))
