from torch.utils.data import Dataset
from torch import as_tensor, Tensor
from typing import Any, Literal

from src.const import DEVICE
from torch import cuda


class TorchDataset(Dataset):
    """A simple PyTorch Dataset wrapper for the AG News dataset."""

    def __init__(self, X: Any, y: Any) -> None:
        """Initialize the dataset with features and labels.

        Args:
            X (Any): The features, needs to be castable to Tensor.
            y (Any): The labels, needs to be castable to Tensor.
        """
        self.X = as_tensor(X)
        self.y = as_tensor(y)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
            tuple[Tensor, Tensor]: A tuple containing the feature tensor and label tensor for the sample.
        """
        return self.X[idx], self.y[idx]

def get_available_vram():
    """Get the available VRAM on the current device."""
    if cuda.is_available():
        return cuda.get_device_properties(DEVICE).total_memory / (1024 ** 3)  # Convert to GB
    else:
        return 0.0