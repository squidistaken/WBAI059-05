from torch import nn
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification
from src.const import HF_TOKEN


class DistilBERTClassifer(nn.Module):
    """DistilBERT Classifer class."""

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        """Initialise the class.

        Args:
            model_name (str, optional): The model name from Hugging Face.
                                        Defaults to "distilbert-base-uncased".
        """
        super(DistilBERTClassifer, self).__init__()

        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=4, token=HF_TOKEN
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Handle forward pass of the DistilBERT classifier.

        Args:
            x (torch.Tensor): The batched input embeddings of shape
                              `(batch_size, seq_len, embedding_dim)`.

        Returns:
            torch.Tensor: The raw logit predictions of shape
                          `(batch_size, num_classes)`.
        """
        input_ids = x[:, :, 0].long()
        attention_mask = x[:, :, 1].long()
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        return outputs.logits

    def predict(
        self, x: torch.Tensor, return_prob: bool = True
    ) -> torch.Tensor:
        """Predict for an input batch.

        Args:
            x (torch.Tensor): The batched input embeddings of shape
                              `(batch_size, seq_len, embedding_dim)`.
            return_prob (bool, optional): Whether to return the class
                                           probabilities with softmax. Defaults
                                           to True.

        Returns:
            torch.Tensor: A tensor containing either the class probabilities
                          of shape `(batch_size, num_classes)` or the predicted
                          class indices of shape `(batch_size,)`.
        """
        self.eval()

        with torch.no_grad():
            logits = self.forward(x)

        if return_prob:
            return F.softmax(logits, dim=1)

        return torch.argmax(logits, dim=1) + 1
