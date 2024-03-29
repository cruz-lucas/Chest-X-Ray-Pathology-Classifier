from transformers import Trainer
from torch import nn
from torch import masked_select


class MaskedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss (masking uncertanty in ignore approach)
        mask = labels > -1.
        criterion = nn.BCEWithLogitsLoss(device=model.device)
        loss = criterion(
            masked_select(logits.view(-1, self.model.config.num_labels), mask),
            masked_select(labels.view(-1, self.model.config.num_labels), mask))
        return (loss, outputs) if return_outputs else loss


class MultiOutputTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss().to(model.device)

        loss_1 = loss_fct(logits[:, 0:3], labels[:, 0:3])
        loss_2 = loss_fct(logits[:, 3:6], labels[:, 3:6])
        loss_3 = loss_fct(logits[:, 6:9], labels[:, 6:9])
        loss_4 = loss_fct(logits[:, 9:12], labels[:, 9:12])
        loss_5 = loss_fct(logits[:, 12:], labels[:, 12:])

        loss = (
            loss_1 + loss_2 + loss_3 + loss_4 + loss_5
        )/5.0
        return (loss, outputs) if return_outputs else loss