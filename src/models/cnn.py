import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional, Dict


class CNNClassifier(nn.Module):
    """CNN Classifier class."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        """Initialize the class.

        Args:
            config (Optional[Dict], optional): The configuration of the CNN.
                                               Defaults to None.
        """
        super(CNNClassifier, self).__init__()

        if config is None:
            config = {}

        # Defaults.
        embedding_dim = config.get("embedding_dim", 100)
        num_classes = config.get("num_classes", 4)
        num_filters = config.get("num_filters", 100)
        filter_sizes = config.get("filter_sizes", [3, 4, 5])
        dropout_rate = config.get("dropout", 0.5)

        # Define layers.
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                )
                for fs in filter_sizes
            ]
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for the CNN"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Handle forward pass of the CNN.

        Args:
            x (torch.Tensor): The batched input embeddings of shape
                              `(batch_size, seq_len, embedding_dim)`.

        Returns:
            torch.Tensor: The raw logit predictions of shape
                          `(batch_size, num_classes)`.
        """
        x = x.permute(0, 2, 1)
        conv_outputs = []

        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled_out = self.pool(conv_out).squeeze(2)
            conv_outputs.append(pooled_out)

        # Concatenate outputs from all filters.
        cat = torch.cat(conv_outputs, dim=1)
        drop = self.dropout(cat)
        logits = self.fc(drop)

        return logits

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
