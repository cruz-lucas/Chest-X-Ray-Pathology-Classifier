import pytorch_lightning as pl

from torchvision.models.efficientnet import efficientnet_v2_l
import torch

from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelPrecisionRecallCurve

class LitEfficientnet(pl.LightningModule):
    def __init__(self,
                 num_classes:int=5,
                 lr=1e-3) -> None:
        super().__init__()
        #self.save_hyperparameters()
        model = efficientnet_v2_l(weights="DEFAULT")
        model.classifier[1] = torch.nn.Linear(1280, num_classes)
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr

        self.auc = MultilabelAUROC(num_labels=num_classes, average=None, thresholds=None)
        self.f1 = MultilabelF1Score(num_labels=num_classes, average=None)
        self.pr_curve = MultilabelPrecisionRecallCurve(num_labels=num_classes, thresholds=None)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        train_loss = self.criterion(output, target)

        train_f1 = self.f1(output, target).mean()

        self.log_dict({"train_loss": train_loss, "train_f1": train_f1}, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        val_loss = self.criterion(output, target)

        target = target.long()

        val_auc = self.auc(output, target).mean()
        #val_precision, val_recall, val_thresholds = self.pr_curve(output, target)
        val_f1 = self.f1(output, target).mean()

        self.log_dict({
             "val_loss": val_loss,
             "val_auc": val_auc,
             #"val_precision": val_precision,
             #"val_recall": val_recall,
             #"val_thresholds": val_thresholds,
             "val_f1": val_f1,
             },
            prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

