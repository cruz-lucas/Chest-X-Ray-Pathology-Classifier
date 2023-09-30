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
        loss = loss_fct(logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss