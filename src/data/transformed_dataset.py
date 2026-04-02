from torch.utils.data import Dataset
import torch
from typing import Callable


class TransformedDataset(Dataset):
    """Dataset class for transformed datasets."""

    def __init__(
        self, original_dataset: Dataset, transform_fn: Callable
    ) -> None:
        """Initialise the class.

        Args:
            original_dataset (Dataset): The original dataset to be transformed.
            transform_fn (function): The transformation function to apply to each item.
        """
        self.original_dataset = original_dataset
        self.transform_fn = transform_fn

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.original_dataset)

    def __getitem__(self, idx: int):
        """Get an item from the dataset and apply the transformation.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            The transformed item.
        """
        x, y = self.original_dataset[idx]
        transformed_item = self.transform_fn(x)
        return transformed_item, y

    @property
    def X(self) -> torch.Tensor:
        """Get the features from the transformed dataset.

        Returns:
            torch.Tensor: The features from the transformed dataset.
        """
        if not hasattr(self, "_X"):
            self._X = torch.stack([self[i][0] for i in range(len(self))])
        return self._X

    @property
    def y(self) -> torch.Tensor:
        """Get the labels from the transformed dataset.

        Returns:
            torch.Tensor: The labels from the transformed dataset.
        """
        if not hasattr(self, "_y"):
            self._y = torch.stack([self[i][1] for i in range(len(self))])
        return self._y
