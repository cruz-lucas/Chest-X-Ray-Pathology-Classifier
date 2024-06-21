"""Module for helper functions."""
from typing import Callable

import torch
from peft import LoraConfig
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelF1Score,
)
from transformers import TrainingArguments, ViTForImageClassification

from chexformer.utils import Constants, TrainerConfig


def get_model(config: TrainerConfig, constants: Constants) -> ViTForImageClassification:
    """Return model.

    Args:
        config (TrainerConfig): Model configuration.
        constants (Constants): Constants of the project.

    Returns:
        ViTForImageClassification: return model for image classification
    """
    n_labels = (
        len(constants.pathologies) * 3 if config.uncertainty_policy == "U-MultiClass" else len(constants.pathologies)
    )

    return ViTForImageClassification.from_pretrained(
        pretrained_model_name_or_path=config.base_model,
        problem_type="multi_label_classification",
        num_labels=n_labels,
        ignore_mismatched_sizes=True,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
    ).to(config.device)


def get_arguments(train_config: TrainerConfig) -> tuple:
    """Get arguments for training and model configuration.

    Args:
        train_config (TrainerConfig): _description_

    Returns:
        tuple: _description_
    """
    training_args = TrainingArguments(
        output_dir=f"./output/{train_config.uncertainty_policy}",
        report_to="mlflow",
        save_strategy="steps",
        save_steps=train_config.save_steps,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        optim="adamw_torch",
        num_train_epochs=train_config.num_epochs,
        learning_rate=train_config.learning_rate,
        lr_scheduler_type=train_config.lr_scheduler_type,
        warmup_ratio=train_config.warmup_ratio,
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation,
        weight_decay=train_config.weight_decay,
        fp16=True if train_config.device == "gpu" else False,
        # max_steps=train_config.max_steps
    )

    lora_args = LoraConfig(
        r=train_config.lora_r,
        lora_alpha=train_config.lora_alpha,
        target_modules=train_config.lora_target_modules,
        lora_dropout=train_config.lora_dropout,
        bias=train_config.lora_bias,
        use_rslora=train_config.lora_use_rslora,
        modules_to_save=["classifier"],
        task_type="multi_label_classification",
    )
    return training_args, lora_args


def prepare_compute_metrics(config: TrainerConfig) -> Callable[[tuple], dict]:
    """Prepare compute metrics for training callback.

    Args:
        config (TrainerConfig): Configuration for training and eval.

    Returns:
        _type_: Return callback function.
    """
    device = config.device

    # Metrics
    AUC = MultilabelAUROC(num_labels=5, average="macro").to(device)
    F1 = MultilabelF1Score(num_labels=5, average="macro").to(device)
    ACC = MultilabelAccuracy(num_labels=5, average="macro").to(device)
    multiclassAUC = MulticlassAUROC(num_classes=3, average="macro").to(device)
    multiclassF1 = MulticlassF1Score(num_classes=3, average="macro").to(device)
    multiclassACC = MulticlassAccuracy(num_classes=3, average="macro").to(device)

    def compute_metrics(eval_pred):
        """Compute metrics for evaluation."""
        nonlocal multiclassAUC, multiclassF1, multiclassACC, device
        logits, labels = eval_pred
        logits = torch.from_numpy(logits).to(device)
        labels = torch.from_numpy(labels).to(device).long()

        if labels.shape[1] == 15:
            auc, f1, acc = 0, 0, 0
            for i in range(0, 15, 3):
                label = torch.argmax(labels[:, i : i + 3], dim=1).int()
                auc += multiclassAUC(logits[:, i : i + 3], label)
                f1 += multiclassF1(logits[:, i : i + 3], label)
                acc += multiclassACC(logits[:, i : i + 3], label)
            auc, f1, acc = auc / 5, f1 / 5, acc / 5
        else:
            auc, f1, acc = AUC(logits, labels), F1(logits, labels), ACC(logits, labels)

        return {
            "auc_roc_mean": auc.cpu().mean().item(),
            "f1_mean": f1.cpu().mean().item(),
            "acc_mean": acc.cpu().mean().item(),
        }

    return compute_metrics
