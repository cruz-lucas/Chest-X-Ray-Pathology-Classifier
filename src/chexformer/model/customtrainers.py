"""Trainers for Ignore and MultiClass approaches."""
from torch import masked_select, nn
from transformers import Trainer


class MaskedLossTrainer(Trainer):
    """Trainer for handling custom loss computation with masking in an ignore approach.

    Args:
        Trainer (transformers.Trainer): Base class for all trainers in Hugging Face Transformers library.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss with masking for uncertain labels.

        Args:
            model (nn.Module): The model being trained.
            inputs (dict): The inputs and targets of the model.
            return_outputs (bool, optional): Whether to return the model outputs in addition to the loss. Defaults to False.

        Returns:
            torch.Tensor or tuple: The computed loss, and optionally the model outputs if return_outputs is True.
        """
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss (masking uncertanty in ignore approach)
        mask = labels > -1.0
        criterion = nn.BCEWithLogitsLoss().to(model.device)
        loss = criterion(
            masked_select(logits.view(-1, self.model.config.num_labels), mask),
            masked_select(labels.view(-1, self.model.config.num_labels), mask),
        )
        return (loss, outputs) if return_outputs else loss


class MultiOutputTrainer(Trainer):
    """Trainer for handling multi-output classification tasks.

    Args:
        Trainer (transformers.Trainer): Base class for all trainers in Hugging Face Transformers library.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss for multi-output classification tasks.

        Args:
            model (nn.Module): The model being trained.
            inputs (dict): The inputs and targets of the model.
            return_outputs (bool, optional): Whether to return the model outputs in addition to the loss. Defaults to False.

        Returns:
            torch.Tensor or tuple: The computed loss, and optionally the model outputs if return_outputs is True.
        """
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

        loss = (loss_1 + loss_2 + loss_3 + loss_4 + loss_5) / 5.0
        return (loss, outputs) if return_outputs else loss
