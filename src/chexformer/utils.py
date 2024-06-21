"""Utility functions and classes."""
from dataclasses import dataclass, field
from typing import List, Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Constants:
    """Object to hold constants types."""

    uncertainty_policies: List[str]
    pathologies: List[str]


@dataclass_json
@dataclass
class TrainerConfig:
    """Object to hold training configs."""

    base_model: Optional[str] = field(default="google/vit-base-patch16-224")
    uncertainty_policy: str = field(default="U-Ignore")
    num_epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 0.00002
    weight_decay: float = 0.02
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    batch_size: int = 8
    gradient_accumulation: int = 1
    save_steps: int = 500
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_use_rslora: bool = True
    device: str = "auto"
    lora_target_modules: List[str] = field(default_factory=lambda: ["q", "v"])
    resume_from_checkpoint: Optional[str] = None


@dataclass_json
@dataclass
class PreprocessConfig:
    """Object to hold preprocessing configs."""

    gcp_project_id: Optional[str]
    gcp_bucket: Optional[str]
    data_path: str = field(default="gs://chexpert_images_database")
    dataset_dir: str = field(default="data/processed/")
    split: str = field(default="train")
    resize: List[int] = field(default_factory=lambda: [224, 224])
