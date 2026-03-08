import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class LSTMClassifier(nn.Module):
    """Class for a v Classifier on NLP."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        """Initialise the LSTM class.

        Args:
            config (Optional[Dict], optional): The configuration of the LSTM.
                                               Defaults to None.
        """

        super(LSTMClassifier, self).__init__()

        if config is None:
            config = {}

        vocab_size = config.get("vocab_size")
        embedding_dim = config.get("embedding_dim", 100)
        hidden_dim = config.get("hidden_dim", 100)
        num_classes = config.get("num_classes", 4)
        num_layers = config.get("num_layers", 4)
        num_filters = config.get("num_filters", 100)
        dropout_rate = config.get("dropout", 0.5)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        rep_dim = hidden_dim * 2
        self.rep_dropout = nn.Dropout(dropout_rate)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(rep_dim, num_classes)        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = x.size()

        packed = nn.utils.rnn.pack_padded_sequence(
            x, batch_first=True, enforce_sorted=False, lengths=lengths
        )

        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]
        drop = self.rep_dropout(h_last)
        logits = self.fc(drop)
        return logits


    def predict(self, x: torch.Tensor, use_softmax: bool = True) -> torch.Tensor:

        self.eval()
        with torch.no_grad():
            logits = self.forward(x)

        if use_softmax:
            return F.softmax(logits, dim=1)
        else:
            return torch.argmax(logits, dim=1) + 1
        

