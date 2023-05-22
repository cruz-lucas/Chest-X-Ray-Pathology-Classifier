import argparse
import os

import gc

from dataset import get_dataloader 
from efficientnet import LitEfficientnet

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch

import wandb

torch.cuda.empty_cache()
gc.collect()

def parse_args():
    parser = argparse.ArgumentParser("train.py")
    parser.add_argument("--dl_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--resize_shape", type=int, default=384)
    parser.add_argument("--input_filepath", type=str)
    parser.add_argument("--uncertainty_policy", type=str, default='U-Ones')
    return parser.parse_args()


def train() -> None:
    args = parse_args()

    with wandb.init(job_type="train", config=args) as run:
        config = run.config

        model = LitEfficientnet(num_classes=5, lr=config.lr)
        wandb_logger = WandbLogger(experiment=run, log_model="all", log_graph=False)

        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = -1
        else:
            accelerator = None
            devices = None

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            default_root_dir=os.getcwd(),
            enable_checkpointing=True,
            callbacks=[
                LearningRateMonitor(),
                ModelCheckpoint(every_n_train_steps=10_000),
                EarlyStopping(monitor="val/loss", mode="min", patience=3, min_delta=0.3, divergence_threshold=0.5)
            ], 
            #precision='bf16-mixed',
            logger=WandbLogger,
            max_epochs=config.max_epochs,
            log_every_n_steps=100,
            enable_progress_bar=True,
            accumulate_grad_batches=config.gradient_accumulation_steps,
            #profiler="advanced"
            )

        wandb_logger.watch(model, log="all", log_freq=1, log_graph=False)

        # Data loader
        train_dataloader = get_dataloader(data_path=config.input_filepath,
                                          uncertainty_policy=config.uncertainty_policy,
                                          train=True,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          num_workers=config.dl_workers,
                                          pin_memory=True,
                                          resize_shape=(config.resize_shape,config.resize_shape))
        valid_dataloader = get_dataloader(data_path=config.input_filepath,
                                          uncertainty_policy=config.uncertainty_policy,
                                          train=False,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          num_workers=config.dl_workers,
                                          pin_memory=True,
                                          resize_shape=(config.resize_shape,config.resize_shape))


        wandb.log({
            "Uncertainty policy": config.uncertainty_policy
        })

        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

        wandb.run.summary["state"] = "completed"
        wandb.finish(quiet=True)


if __name__ == '__main__':
    train()