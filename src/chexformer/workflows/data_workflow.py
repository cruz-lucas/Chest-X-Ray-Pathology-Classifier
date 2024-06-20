"""Data preprocessing workflow."""
import json

from chexformer.data import CheXpertDataset
from chexformer.utils import PreprocessConfig


def preprocess_dataset_workflow(config: PreprocessConfig):
    """Run preprocess workflow and persists dataset locally.

    Args:
        config (PreprocessConfig): Configuration for preprocessing data.
    """
    dataset = CheXpertDataset(config=config)
    dataset.load_from_raw_data()
    dataset.preprocess_dataset()


if __name__ == "__main__":
    with open("./src/chexformer/config/preprocess_config.json") as f:
        json_obj = json.load(f)
    preprocess_dataset_workflow(config=PreprocessConfig(**json_obj))
