"""Utility functions and classes."""
from dataclasses import dataclass, field
from typing import List, Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class TrainerConfig:
    """Object to hold training configs."""

    base_model: Optional[str] = field(default="EleutherAI/pythia-1B-deduped")
    data_path: str = field(default="yahma/alpaca-cleaned", metadata={"help": "Path to the training data."})
    data_name: str = field(default="chexpert", metadata={"help": "Path to the training data config name."})
    num_epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 0.00002
    weight_decay: float = 0.02
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    batch_size: int = 8
    micro_batch_size: int = 1
    val_set_size: int = 0
    group_by_length: bool = False
    instruction_key: str = "instruction"
    input_key: str = "input"
    output_key: str = "output"
    device_map: str = "auto"
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    debug_mode: bool = False
    debug_train_data_size: int = 1024


@dataclass_json
@dataclass
class Constants:
    """Object to hold constants types."""

    uncertainty_policies: List[str]
    pathologies: List[str]


@dataclass_json
@dataclass
class PreprocessConfig:
    """Object to hold preprocessing configs."""

    gcp_project_id: Optional[str]
    gcp_bucket: Optional[str]
    constants: Constants
    data_path: str = field(default="gs://chexpert_images_database")
    dataset_path: str = field(default="data/processed/chexpert.tar")
    split: str = field(default="train")
    resize: List[int] = field(default_factory=lambda: [224, 224])
