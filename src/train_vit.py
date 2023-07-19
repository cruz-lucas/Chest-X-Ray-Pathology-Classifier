import gc
import logging
import os
import io
from typing import List, Union
import argparse
import yaml

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelPrecisionRecallCurve,
    MultilabelAccuracy
)

from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)

import wandb

from google.cloud import storage

import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

torch.cuda.empty_cache()
gc.collect()

# CheXpert pathologies on original paper
pathologies = ['Atelectasis',
               'Cardiomegaly',
               'Consolidation',
               'Edema',
               'Pleural Effusion']

# Uncertainty policies on original paper
uncertainty_policies = ['U-Ignore',
                        'U-Zeros',
                        'U-Ones',
                        'U-SelfTrained',
                        'U-MultiClass']

######################
## Create a Dataset ##
######################
class CheXpertDataset(Dataset):
    def __init__(self,
                 data_path: Union[str, None] = None,
                 uncertainty_policy: str = 'U-Ones',
                 logger: logging.Logger = logging.getLogger(__name__),
                 pathologies: List[str] = pathologies,
                 train: bool = True,
                 resize_shape: tuple = (256, 256)) -> None:
        """ Innitialize dataset and preprocess according to uncertainty policy.

        Args:
            data_path (str): Path to csv file.
            uncertainty_policy (str): Uncertainty policies compared in the original paper.
            Check if options are implemented. Options: 'U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', and 'U-MultiClass'.
            logger (logging.Logger): Logger to log events during training.
            pathologies (List[str], optional): Pathologies to classify.
            Defaults to 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', and 'Pleural Effusion'.
            transform (type): method to transform image.
            train (bool): If true, returns data selected for training, if not, returns data selected for validation (dev set), as the CheXpert research group splitted.

        Returns:
            None
        """
        
        if not(uncertainty_policy in uncertainty_policies):
            logger.error(f"Unknown uncertainty policy. Known policies: {uncertainty_policies}")
            return None

        split = 'train' if train  else 'valid'
        csv_path = f"CheXpert-v1.0/{split}.csv"
        path = str(data_path) + csv_path

        self.in_cloud = False

        data = pd.DataFrame()
        try:
            data = pd.read_csv(path)
            data['Path'] = data_path + data['Path']
            logger.info("Local database found.")
        except Exception as e:
            try:
              ### Find files at gcp
                project_id = 'labshurb'

                storage_client = storage.Client(project=project_id)
                self.bucket = storage_client.bucket('chexpert_database_stanford')

                blob = self.bucket.get_blob(csv_path)
                blob.download_to_filename('tmp.csv')
                data = pd.read_csv('tmp.csv')

                self.in_cloud = True
                logger.info("Cloud database found.")
            except:  
                logger.error(f"Couldn't read csv at path {path}./n{e}")
                quit()

        data.set_index('Path', inplace=True)

        #data = data.loc[data['Frontal/Lateral'] == 'Frontal'].copy()
        data = data.loc[:, pathologies].copy()
        
        data.fillna(0, inplace=True)

        # U-Ignore
        if uncertainty_policy == uncertainty_policies[0]:
            data = data.loc[(data[pathologies] != -1).all(axis=1)].copy()
        
        # U-Zeros
        elif uncertainty_policy == uncertainty_policies[1]:
            data.replace({-1: 0}, inplace=True)

        # U-Ones
        elif uncertainty_policy == uncertainty_policies[2]:
            data.replace({-1: 1}, inplace=True)

        # U-SelfTrained
        elif uncertainty_policy == uncertainty_policies[3]:
            logger.error(f"Uncertainty policy {uncertainty_policy} not implemented.")
            return None

        # U-MultiClass
        elif uncertainty_policy == uncertainty_policies[4]:
            data.replace({-1: 2}, inplace=True)

        self.image_names = data.index.to_numpy()
        self.labels = data.loc[:, pathologies].to_numpy()
        self.transform = T.Compose([
                  T.Resize(resize_shape),
                  T.ToTensor(),
                  T.Normalize(mean=[0.5330], std=[0.0349])
              ]) # whiten with dataset mean and stdif transform)

    def __getitem__(self, index: int) -> Union[np.array, Tensor]:
        """ Returns image and label from given index.

        Args:
            index (int): Index of sample in dataset.

        Returns:
            np.array: Array of grayscale image.
            torch.Tensor: Tensor of labels.
        """
        if self.in_cloud:
            img_bytes = self.bucket.blob(self.image_names[index]).download_as_bytes()#.download_to_filename('tmp.jpg')
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        else:
            img = Image.open(self.image_names[index]).convert('RGB')
        img = self.transform(img)

        label = self.labels[index].astype(np.float32)
        return {"pixel_values": img, "labels": label}

    def __len__(self) -> int:
        """ Return length of dataset.

        Returns:
            int: length of dataset.
        """
        return len(self.image_names)


def get_args():
    '''Parses args.'''

    parser = argparse.ArgumentParser("train_vit.py")
    parser.add_argument(
        "--sweep_config",
        type=str,
        default='src/config.yaml',
        help="Path to yaml file containing sweep config"
    )
    parser.add_argument(
        '--job_dir',
        required=False,
        type=str,
        default='.',
        help='Bucket to store saved model, include gs://')
    parser.add_argument(
        '--data_path',
        required=False,
        type=str,
        default=r"C:/Users/hurbl/OneDrive/Área de Trabalho/Loon Factory/repository/Chest-X-Ray-Pathology-Classifier/data/raw/",
        #default="gcs://chexpert_database_stanford/",
        help='Local or storage path to csv metadata file'
    )
    parser.add_argument(
        '--uncertainty_policy',
        required=False,
        type=str,
        default=uncertainty_policies[-1],
        help='Uncertainty policy'
    )
    args = parser.parse_args()
    return args


AUC = MultilabelAUROC(num_labels=5, average=None, thresholds=None).to('cuda')
F1 = MultilabelF1Score(num_labels=5, average=None).to('cuda')
PR_CURVE = MultilabelPrecisionRecallCurve(num_labels=5, thresholds=None).to('cuda')
ACC = MultilabelAccuracy(num_labels=5, average=None).to('cuda')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.from_numpy(logits).to('cuda')
    labels = torch.from_numpy(labels).to('cuda').long()

    auc = AUC(logits, labels)
    #precision, recall, _ = PR_CURVE(logits, labels)
    f1 = F1(logits, labels)
    acc = ACC(logits, labels)

    return {
        #'auc_roc': auc,
        #'precision': precision,
        #'recall': recall,
        #'f1': f1,
        'auc_roc_mean': auc.cpu().mean(),
        #'precision_mean': np.mean(precision),
        #'recall_mean': np.mean(recall),
        'f1_mean': f1.cpu().mean(),
        'acc_mean': acc.cpu().mean()
    }


def main(args):
    with wandb.init(job_type="train") as run:
        config = run.config

        train_dataset = CheXpertDataset(
            data_path=args.data_path,
            uncertainty_policy=args.uncertainty_policy,
            train=True,
            resize_shape=(384, 384))


        val_dataset = CheXpertDataset(
            data_path=args.data_path,
            uncertainty_policy=args.uncertainty_policy,
            train=False,
            resize_shape=(384, 384))

        model_ckp = "google/vit-large-patch16-384"        
        model = ViTForImageClassification.from_pretrained(
            model_ckp, 
            problem_type="multi_label_classification",
            num_labels=5,
            ignore_mismatched_sizes=True
        )

        training_args = TrainingArguments(
                output_dir=f"./output/{model_ckp}/{args.uncertainty_policy}",
                report_to='wandb',  # Turn on Weights & Biases logging
                save_strategy='epoch',
                evaluation_strategy="epoch",
                logging_strategy='steps',
                logging_steps=1,
                optim='adamw_torch',
                num_train_epochs=config.epochs,
                learning_rate=config.lr,
                lr_scheduler_type='linear',
                warmup_steps=1_000,
                max_grad_norm=1.0,
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.grad_acc,
                weight_decay=0.1,
                #gradient_checkpointing=True,
                auto_find_batch_size=False,
                fp16=True,
                dataloader_drop_last=True,
                load_best_model_at_end=True,
            )

        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
            )

        train_results = trainer.train()
        trainer.save_model(f'{args.job_dir}/{args.uncertainty_policy}/model_output')

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
    with open(args.sweep_config) as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)

    wandb.agent(sweep_id=sweep_id, function=(lambda: main(args=args)), count=1)