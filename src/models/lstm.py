import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class LSTMClassifier(nn.Module):
    """LSTM Classifier class."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        """Initialise the LSTM class.

        Args:
            config (Optional[Dict], optional): The configuration of the LSTM.
                                               Defaults to None.
        """

        super(LSTMClassifier, self).__init__()

        if config is None:
            config = {}

        # Defaults.
        embedding_dim = config.get("embedding_dim", 100)
        hidden_dim = config.get("hidden_dim", 100)
        num_classes = config.get("num_classes", 4)
        num_layers = config.get("num_layers", 4)
        dropout_rate = config.get("dropout", 0.5)

        # Define layers.
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Handle forward pass of the LSTM.

        Args:
            x (torch.Tensor): The batched input embeddings of shape
                              `(batch_size, seq_len, embedding_dim)`.

        Returns:
            torch.Tensor: The raw logit predictions of shape
                          `(batch_size, num_classes)`.
        """
        x = self.dropout(x)
        output, (h_n, c_n) = self.lstm(x)

        # Concatenate the final hidden states from both directions.
        h_last = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        drop = self.dropout(h_last)
        out = self.fc(drop)

        return out

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
