"""Train script for CheXpert with Vision Transformer."""
from datetime import datetime

from transformers import (
    ViTForImageClassification, TrainingArguments, Trainer
)
import wandb

from chexpert import CheXpertDataset
from custom_trainer import MaskedLossTrainer, MultiOutputTrainer

import torch
from torchmetrics.classification import (
    MultilabelAUROC, MultilabelF1Score, MultilabelAccuracy,
    MulticlassAUROC, MulticlassF1Score, MulticlassAccuracy
)

import logging
import gc

# Set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.empty_cache()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure garbage collection
gc.collect()

# Metrics
AUC = MultilabelAUROC(num_labels=5, average='macro').to(device)
F1 = MultilabelF1Score(num_labels=5, average='macro').to(device)
ACC = MultilabelAccuracy(num_labels=5, average='macro').to(device)
multiclassAUC = MulticlassAUROC(num_classes=3, average='macro').to(device)
multiclassF1 = MulticlassF1Score(num_classes=3, average='macro').to(device)
multiclassACC = MulticlassAccuracy(num_classes=3, average='macro').to(device)


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    logits = torch.from_numpy(logits).to(device)
    labels = torch.from_numpy(labels).to(device).long()

    if labels.shape[1] == 15:
        auc, f1, acc = 0, 0, 0
        for i in range(0, 15, 3):
            label = torch.argmax(labels[:, i:i+3], dim=1).int()
            auc += multiclassAUC(logits[:, i:i+3], label)
            f1 += multiclassF1(logits[:, i:i+3], label)
            acc += multiclassACC(logits[:, i:i+3], label)
        auc, f1, acc = auc / 5, f1 / 5, acc / 5
    else:
        auc, f1, acc = AUC(logits, labels), F1(logits, labels), ACC(logits, labels)

    return {
        'auc_roc_mean': auc.cpu().mean().item(),
        'f1_mean': f1.cpu().mean().item(),
        'acc_mean': acc.cpu().mean().item()
    }


def main():
    """Main function to run the training script."""
    config = get_config()

    with wandb.init(
        project=config['wandb']['project'],
        job_type=config['wandb']['job_type'],
        config=config,
        name=f"{config['trainer']['uncertainty_policy']}_{datetime.now().strftime('%d%m%Y_%H%M%S')}",
        tags=[config['trainer']['uncertainty_policy'], config['trainer']['checkpoint']]
    ):
        train_dataset = CheXpertDataset(
            data_path=config.preprocess.data_path,
            uncertainty_policy=config.trainer.uncertainty_policy,
            train=True,
            csv_name=config.split,
            resize_shape=config.trainer.resize
        )

        val_dataset = CheXpertDataset(
            data_path=config.preprocess.data_path,
            uncertainty_policy=config.trainer.uncertainty_policy,
            train=False,
            resize_shape=config.trainer.resize
        )

        num_labels = 15 if config.trainer.uncertainty_policy == 'U-MultiClass' else 5

        model = ViTForImageClassification.from_pretrained(
            config.trainer.checkpoint,
            problem_type="multi_label_classification",
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            attn_implementation="sdpa",
            torch_dtype=torch.float16
        ).to(device)

        training_args = TrainingArguments(
            output_dir=f"./output/{datetime.now().strftime('%Y%m%d')}/{config.trainer.checkpoint}/{config.trainer.uncertainty_policy}",
            report_to='wandb',
            save_strategy='steps',
            save_steps=0.05,
            evaluation_strategy="epoch",
            logging_strategy='steps',
            logging_steps=1,
            optim='adamw_torch',
            num_train_epochs=config.trainer.epochs,
            learning_rate=config.trainer.learning_rate,
            lr_scheduler_type='linear',
            warmup_steps=1000,
            max_grad_norm=1.0,
            per_device_train_batch_size=config.trainer.batch_size,
            gradient_accumulation_steps=config.trainer.gradient_accumulation,
            weight_decay=0.1,
            fp16=True,
            dataloader_drop_last=True,
            push_to_hub=True,
            hub_strategy='checkpoint',
            hub_private_repo=False,
            hub_model_id=f"{config.checkpoint}-CheXpert-{config.trainer.uncertainty_policy}"
        )

        trainer_cls = {
            'U-Ignore': MaskedLossTrainer,
            'U-MultiClass': MultiOutputTrainer
        }.get(config.trainer.uncertainty_policy, Trainer)

        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        train_results = trainer.train()

        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
