import gc
import os
import argparse
from datetime import datetime

import torch
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelAccuracy
)

from chexpert import CheXpertDataset
from custom_trainer import MaskedLossTrainer, MultiOutputTrainer

from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)

import wandb

import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

gc.collect()

# Uncertainty policies on original paper
uncertainty_policies = ['U-Ignore',
                        'U-Zeros',
                        'U-Ones',
                        'U-SelfTrained',
                        'U-MultiClass']


device = 'cpu'
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = 'cuda'


def get_args():
    '''Parses args.'''

    parser = argparse.ArgumentParser("train_vit.py")
    parser.add_argument(
        "--epochs",
        "-e",
        required=False,
        type=int,
        default=5,
        help="Epochs of training"
    )
    parser.add_argument(
        "--learning_rate",
        "-l",
        required=False,
        type=float,
        default=4e-4,
        help="learning rate of training"
    )
    parser.add_argument(
        "--gradient_accumulation",
        "-g",
        required=False,
        type=int,
        default=64,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        required=False,
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        '--job_dir',
        '-j',
        required=False,
        type=str,
        default='.',
        help='Bucket to store saved model, include gs://')
    parser.add_argument(
        '--data_path',
        '-d',
        required=False,
        type=str,
        default=r"C:/Users/hurbl/OneDrive/Área de Trabalho/Loon Factory/repository/Chest-X-Ray-Pathology-Classifier/data/raw/",
        # default="gcs://chexpert_database_stanford/",
        help='Local or storage path to csv metadata file' 
    )
    parser.add_argument(
        '--uncertainty_policy',
        '-u',
        required=False,
        type=str,
        default=uncertainty_policies[0],
        help='Uncertainty policy'
    )
    parser.add_argument(
        '--resize',
        '-r',
        required=False,
        type=tuple,
        default=(224, 224),
        help='Resize dimensions'
    )
    parser.add_argument(
        '--checkpoint',
        '-c',
        required=False,
        type=str,
        default='google/vit-base-patch16-224',
        help='checkpoint to load from hugging face hub'
    )
    args = parser.parse_args()
    return args


AUC = MultilabelAUROC(num_labels=5, average='macro', thresholds=None).to(device)
F1 = MultilabelF1Score(num_labels=5, average='macro').to(device)
ACC = MultilabelAccuracy(num_labels=5, average='macro').to(device)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.from_numpy(logits).to(device)
    labels = torch.from_numpy(labels).to(device).long()

    auc = AUC(logits, labels)
    f1 = F1(logits, labels)
    acc = ACC(logits, labels)

    return {
        'auc_roc_mean': auc.cpu().mean(),
        'f1_mean': f1.cpu().mean(),
        'acc_mean': acc.cpu()
    }


def main(args):
    with wandb.init(project="chexpert-vit", job_type="train", config=args,
                    name=str(args.uncertainty_policy)+str(datetime.now().strftime("%d%m%Y_%H%M%S")),
                    tags=[
                        args.uncertainty_policy,
                        args.checkpoint]) as run:
        config = run.config

        train_dataset = CheXpertDataset(
            data_path=config['data_path'],
            uncertainty_policy=config['uncertainty_policy'],
            train=True,
            resize_shape=config['resize'])

        val_dataset = CheXpertDataset(
            data_path=config['data_path'],
            uncertainty_policy=config['uncertainty_policy'],
            train=False,
            resize_shape=config['resize'])

        num_labels = 15 if config['uncertainty_policy'] == 'U-MultiClass' else 5

        model = ViTForImageClassification.from_pretrained(
            config['checkpoint'], 
            problem_type="multi_label_classification",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(device)

        training_args = TrainingArguments(
                output_dir=f"./output/25092023/{config['checkpoint']}/{config['uncertainty_policy']}",
                report_to='wandb',  # Turn on Weights & Biases logging
                save_strategy='steps',
                save_steps=0.05,
                evaluation_strategy="epoch",
                logging_strategy='steps',
                logging_steps=1,
                optim='adamw_torch',
                num_train_epochs=config['epochs'],
                learning_rate=config['learning_rate'],
                lr_scheduler_type='linear',
                warmup_steps=1_000,
                max_grad_norm=1.0,
                per_device_train_batch_size=config['batch_size'],
                gradient_accumulation_steps=config['gradient_accumulation'],
                weight_decay=0.1,
                # gradient_checkpointing=True,
                auto_find_batch_size=False,
                #fp16=True,
                dataloader_drop_last=True,
                #load_best_model_at_end=True,
                push_to_hub=True,
                hub_strategy='checkpoint',
                hub_private_repo=False,
                hub_model_id=f"lucascruz/CheXpert-ViT-{config['uncertainty_policy']}",
            )

        if config['uncertainty_policy'] == 'U-Ignore':
            trainer = MaskedLossTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                )
        elif config['uncertainty_policy'] == 'U-MultiClass':
            trainer = MultiOutputTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                )
        else:
            trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=compute_metrics,
                )

        train_results = trainer.train()
        # trainer.save_model(f'{config["job_dir"]}/{config["uncertainty_policy"]}/model_output')

        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        metrics = trainer.evaluate()
        # some nice to haves:
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    project_name = "chexpert-vit"
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_LOG_MODEL"] = "true"

    args = get_args()
    main(args)