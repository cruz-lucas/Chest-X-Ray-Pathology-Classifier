 import click
import logging
from tqdm import tqdm
import gc
import time
import numpy as np

from src.data.dataset import get_dataloader
from src.models.utils import load_checkpoint, save_checkpoint

import torch
import torch.optim as optim
from torchvision.models.efficientnet import efficientnet_v2_s

from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score

import wandb

torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)

# Data paths default values
RAW_DATA_PATH = r"/media/lucas/Lucas' Backup Disk/"
CHECKPOINT_PATH = 'models/ckpt/'

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
@click.option('--uncertainty_policy', '-u', type=str,
    help='Policy to handle uncertainty.According the CheXpert original paper, policies are "U-Ignore", "U-Zeros", "U-Ones", "U-SelfTrained", and "U-MultiClass".')
def train(input_filepath: str,
          uncertainty_policy: str) -> None:

    logger = logging.getLogger(__name__)   
    gc.collect() 

    project_name = 'Chest-X-Ray-Pathology-Classifier'
    wandb.init(
        project=project_name,
        entity="lucas_cruz",
        group="hp_optimization",
        reinit=True,
    )

    # Hyperparameters
    BATCH_SIZE = wandb.config.batch_size
    OPTIM_NAME = wandb.config.optimizer
    RESIZE_SHAPE = (320,320)
    DEVICE, PIN_MEMORY = get_device()
    LEARNING_RATE = wandb.config.lr
    EPOCHS = wandb.config.epochs
    NUM_WORKERS = 1
    NUM_CLASSES = 5


    # Fetch model
    model = efficientnet_v2_s(num_classes=NUM_CLASSES).to(DEVICE)
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

    for epoch in range(EPOCHS):
        train_epoch(optimizer, model, train_data_loader, DEVICE, criterion, NUM_CLASSES, BATCH_SIZE)
        val_auc = validate(model, valid_data_loader, DEVICE, criterion, NUM_CLASSES, BATCH_SIZE)


    wandb.run.summary["final_auc"] = val_auc
    wandb.run.summary["state"] = "completed"
    wandb.finish(quiet=True)

    return val_auc


def train_epoch(optimizer, model, train_loader, device, criterion, num_classes, batch_size):
    model.train()
    
    outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.size(dim=0) != batch_size:
            continue

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        outputs = torch.cat((outputs, output.detach()), 0)
        targets = torch.cat((targets, target.detach()), 0)
        
        wandb.log({
            "batch_loss": loss.item(),
        })

    auc = MultilabelAUROC(num_labels=num_classes).to(device)(outputs, targets)
    f1 = MultilabelF1Score(num_labels=num_classes).to(device)(outputs, targets)

    wandb.log({
        "training AUC": auc,
        "training F1": f1,
    })


def validate(model, val_loader, device, criterion, num_classes, batch_size):
    model.eval()

    outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            if data.size(dim=0) != batch_size:
                continue
            data, target = data.to(device), target.to(device)
            output = model(data)

            outputs = torch.cat((outputs, output.detach()), 0)
            targets = torch.cat((targets, target.detach()), 0)

        auc = MultilabelAUROC(num_labels=num_classes).to(device)(outputs, targets)
        f1 = MultilabelF1Score(num_labels=num_classes).to(device)(outputs, targets)

        wandb.log({
            "valid AUC": auc,
            "valid F1": f1,
        })

    return auc


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    train()
