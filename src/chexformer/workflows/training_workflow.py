"""Data preprocessing workflow."""
import json

from chexformer.data import CheXpertDataset
from chexformer.utils import PreprocessConfig, TrainerConfig


def train_workflow(data_config: PreprocessConfig, train_config: TrainerConfig):
    """Run preprocess workflow and persists dataset locally.

    Args:
        data_config (PreprocessConfig): Configuration for loading data.
        train_config (TrainerConfig): Configuration for training model.
    """
    dataset = CheXpertDataset(config=data_config).load_from_webdataset()
    dataset = dataset.load_from_webdataset()


if __name__ == "__main__":
    with open("./src/chexformer/config/preprocessing_config.json") as f:
        data_json_obj = json.load(f)
    with open("./src/chexformer/config/training_config.json") as f:
        train_json_obj = json.load(f)
    train_workflow(data_config=PreprocessConfig(**data_json_obj), train_config=TrainerConfig(**train_json_obj))
