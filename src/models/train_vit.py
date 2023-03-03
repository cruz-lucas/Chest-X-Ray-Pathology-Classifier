import click
import logging
from tqdm import tqdm
import gc
import time
import numpy as np
import os

from src.data.dataset import get_dataloader

import torch
import torch.optim as optim
from transformers import AutoModelForImageClassification

from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score

import wandb

torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)

# Data paths default values
RAW_DATA_PATH = r"D:/"
CHECKPOINT_PATH = 'models/ckpt/'

# method
sweep_config = {
    'method': 'bayes'
}

# early stop
early_stop = {
    'type': 'hyperband',
    'min_iter': 3
}

# hyperparameters
parameters_dict = {
    'epochs': {
        'values': [10, 15]
        },
    'batch_size': {
        'values': [16, 32, 64]
        },
    'lr': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-3
    },
    'weight_decay': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    },
}

# metric
sweep_metric = {
    'name': 'valid AUC',
    'goal': 'maximize'
}

def get_device():
    device = torch.device('cpu')
    pin_memory = False

    if torch.cuda.is_available():
        device = torch.device('cuda')
        pin_memory = True
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)

    else:
        print("warning! GPU not available.")

    return device, pin_memory


@click.command()
@click.option('--input_filepath', '-i', default=RAW_DATA_PATH, type=str, help='Path to input data.')
@click.option('--uncertainty_policy', '-u',  default='U-Ones', type=str,
    help='Policy to handle uncertainty.According the CheXpert original paper, policies are "U-Ignore", "U-Zeros", "U-Ones", "U-SelfTrained", and "U-MultiClass".')
def train(input_filepath: str,
          uncertainty_policy: str,
          config = None) -> None:

    logger = logging.getLogger(__name__)   
    gc.collect() 

    with wandb.init(config=config):
        # set sweep configuration
        config = wandb.config

        # Hyperparameters
        BATCH_SIZE = wandb.config.batch_size
        OPTIM_NAME = "Adam"
        RESIZE_SHAPE = (224,224)
        DEVICE, PIN_MEMORY = get_device()
        LEARNING_RATE = wandb.config.lr
        EPOCHS = wandb.config.epochs
        NUM_WORKERS = 1
        NUM_CLASSES = 5


        # Fetch model
        checkpoint = "google/vit-base-patch16-224-in21k"

        labels = [
            'Atelectasis',
            'Cardiomegaly',
            'Consolidation',
            'Edema',
            'Pleural Effusion'
        ]

        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        model = AutoModelForImageClassification.from_pretrained(
            checkpoint,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        ).to(DEVICE)
        # create losses (criterion in pytorch)
        criterion = torch.nn.CrossEntropyLoss()

        wandb.watch(model, criterion=criterion, log="all", log_freq=1)

        # Data loader
        train_data_loader = get_dataloader(data_path=input_filepath,
                                        uncertainty_policy=uncertainty_policy,
                                        logger=logger,
                                        train=True,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY,
                                        resize_shape=RESIZE_SHAPE)
        valid_data_loader = get_dataloader(data_path=input_filepath,
                                        uncertainty_policy=uncertainty_policy,
                                        logger=logger,
                                        train=False,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY,
                                        resize_shape=RESIZE_SHAPE)


        # Optimizer
        optimizer = getattr(optim, OPTIM_NAME)(model.parameters(), lr=LEARNING_RATE)
        scaler = torch.cuda.amp.GradScaler()

        wandb.log({
            "Uncertainty policy": uncertainty_policy
        })

        for epoch in range(EPOCHS):
            train_epoch(optimizer, scaler, model, train_data_loader, DEVICE, criterion, NUM_CLASSES, BATCH_SIZE, epoch)
            val_auc = validate(model, valid_data_loader, DEVICE, criterion, NUM_CLASSES, BATCH_SIZE, epoch)

        wandb.run.summary["final auc"] = val_auc
        wandb.run.summary["state"] = "completed"
        wandb.finish(quiet=True)

        #return val_auc


def train_epoch(optimizer, scaler, model, train_loader, device, criterion, num_classes, batch_size, epoch):
    model.train()
    
    outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.size(dim=0) != batch_size:
            continue

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Automatic Tensor Casting
        with torch.cuda.amp.autocast():
            output = model(data).logits
            loss = criterion(output, target)

        # Automatic Gradient Scaling
        scaler.scale(loss).backward()

        # Update Optimizer
        scaler.step(optimizer)
        scaler.update()

        outputs = torch.cat((outputs, output.detach()), 0)
        targets = torch.cat((targets, target.detach()), 0)
        
        wandb.log({
            "batch_loss": loss.item(),
            "batch": batch_idx,
            "epoch": epoch
        })

        # Garbage Collection
        torch.cuda.empty_cache()
        _ = gc.collect()


    auc = MultilabelAUROC(num_labels=num_classes).to(device)(outputs, targets.to(torch.int32))
    f1 = MultilabelF1Score(num_labels=num_classes).to(device)(outputs, targets.to(torch.int32))

    wandb.log({
        "training AUC": auc,
        "training F1": f1,
        "epoch": epoch
    })


def validate(model, val_loader, device, criterion, num_classes, batch_size, epoch):
    model.eval()

    outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            if data.size(dim=0) != batch_size:
                continue
            data, target = data.to(device), target.to(device)
            output = model(data).logits

            outputs = torch.cat((outputs, output.detach()), 0)
            targets = torch.cat((targets, target.detach()), 0)

        auc = MultilabelAUROC(num_labels=num_classes).to(device)(outputs, targets.to(torch.int32))
        f1 = MultilabelF1Score(num_labels=num_classes).to(device)(outputs, targets.to(torch.int32))

        wandb.log({
            "valid AUC": auc,
            "valid F1": f1,
            "epoch": epoch
        })

    return auc


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    wandb.login()

    os.environ["WANDB_PROJECT"] = "Chest-X-Ray-Pathology-Classifier"
    os.environ["WANDB_LOG_MODEL"] = "true"

    sweep_config['parameters'] = parameters_dict
    sweep_config['early_terminate'] = early_stop
    sweep_config['metric'] = sweep_metric

    sweep_id = wandb.sweep(sweep_config, project='Chest-X-Ray-Pathology-Classifier')
    #sweep_id = 'mtafcuyn'

    wandb.agent(sweep_id, train, count=20)
