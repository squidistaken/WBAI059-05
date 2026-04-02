from torch.utils.data import Dataset
from torch import as_tensor, Tensor
from typing import Any
from src.const import DEVICE
from torch import cuda


class TorchDataset(Dataset):
    """PyTorch Dataset wrapper class for the AGNews dataset."""

    def __init__(self, X: Any, y: Any) -> None:
        """Initialize the dataset with features and labels.

        Args:
            X (Any): The features to be castable as a Tensor.
            y (Any): The labels to be castable as a Tensor.
        """
        self.X = as_tensor(X)
        self.y = as_tensor(y)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the feature tensor and
                                   label tensor for the sample.
        """
        return self.X[idx], self.y[idx]


def get_available_vram():
    """Get the available VRAM on the current device."""
    if cuda.is_available():
        # Convert to GB
        return cuda.get_device_properties(DEVICE).total_memory / (1024**3)

    return 0.0
