"""Data preprocessing workflow."""
import json

from peft import get_peft_model
from transformers import Trainer

from chexformer.data import CheXpertDataset
from chexformer.model.customtrainers import MaskedLossTrainer, MultiOutputTrainer
from chexformer.model.helpers import get_arguments, get_model, prepare_compute_metrics
from chexformer.utils import Constants, PreprocessConfig, TrainerConfig


def train_workflow(data_config: PreprocessConfig, train_config: TrainerConfig, constants: Constants):
    """Run preprocess workflow and persists dataset locally.

    Args:
        data_config (PreprocessConfig): Configuration for loading data.
        train_config (TrainerConfig): Configuration for training model.
        constants (Constants): Constants used in the project.
    """
    dataset = CheXpertDataset(config=data_config, constants=constants)
    train_dataset = dataset.load_from_webdataset(split="train")
    valid_dataset = dataset.load_from_webdataset(split="valid")

    training_args, lora_args = get_arguments(train_config)
    model = get_model(config=train_config, constants=constants)
    lora_model = get_peft_model(model=model, peft_config=lora_args)

    trainer_cls = {"U-Ignore": MaskedLossTrainer, "U-MultiClass": MultiOutputTrainer}.get(
        train_config.uncertainty_policy, Trainer
    )

    trainer = trainer_cls(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=prepare_compute_metrics(train_config),
    )

    train_results = trainer.train()

    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    with open("./src/chexformer/config/constants.json") as f:
        const_json_obj = json.load(f)
    with open("./src/chexformer/config/preprocess_config.json") as f:
        data_json_obj = json.load(f)
    with open("./src/chexformer/config/local_training_config.json") as f:
        train_json_obj = json.load(f)
    train_workflow(
        data_config=PreprocessConfig(**data_json_obj),
        train_config=TrainerConfig(**train_json_obj),
        constants=Constants(**const_json_obj),
    )
