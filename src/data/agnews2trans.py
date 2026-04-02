from src.data.agnews import AGNews
from src.utils.singleton import SingletonMeta
from transformers import DistilBertTokenizerFast
from src.utils.data import TorchDataset
from src.data.transformed_dataset import TransformedDataset
from src.const import LOGGER, HF_TOKEN
import torch
import torch.nn.functional as F
from rich.panel import Panel
from torch.utils.data import Dataset
from typing import Callable, Literal
from torch import Tensor


class AGNews2Trans(AGNews, metaclass=SingletonMeta):
    """Transformer Dataset wrapper class for the AG News dataset."""

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        super().__init__()

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            model_name, token=HF_TOKEN
        )

    def get_torch_dataset(
        self,
        split: str,
        max_length: int = 256,
        transform_fn: Callable | None = None,
    ) -> TorchDataset:
        """Get a TorchDataset for a given split.

        Args:
            split (str): The split.
            max_length (int, optional): The maximum sequence length. Defaults
                                        to 256.

        Raises:
            ValueError: If an invalid split name is used.

        Returns:
            TorchDataset: A TorchDataset object.
        """
        if split == "train":
            df = self.train_df
        elif split == "dev":
            df = self.dev_df
        elif split == "test":
            df = self.test_df
        else:
            raise ValueError(
                "Invalid split name. Use 'train', 'dev', or 'test'."
            )

        texts = df["text"].to_list()

        LOGGER.log_and_print(
            Panel(
                f"Preloading DistilBERT Dataset for the {split.capitalize()} Split...",
                style="bold yellow",
            )
        )

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        X = torch.stack(
            [encodings["input_ids"], encodings["attention_mask"]], dim=-1
        )

        y = F.one_hot(
            torch.tensor(df["label"].to_numpy() - 1), num_classes=4
        ).float()

        tds = TorchDataset(X, y)

        if transform_fn is not None:
            tds = TransformedDataset(tds, transform_fn)

        return tds


class AGNews2TransDataset(Dataset):
    """PyTorch Dataset wrapper class for the AG News dataset with Word2Trans
    tokenisation."""

    def __init__(
        self,
        split: Literal["train", "dev", "test"] = "train",
        max_length: int = 256,
    ) -> None:
        self.ds = AGNews2Trans()
        self.df = {
            "train": self.ds.train_df,
            "dev": self.ds.dev_df,
            "test": self.ds.test_df,
        }[split]
        self.tokenizer = self.ds.tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        """Get the total number of samples in the split.

        Returns:
            int: The total number of samples in the split.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple[Tensor, Tensor]: A tuple `(embedding_tensor, one_hot_label)`.
        """
        text = self.df["text"][idx]
        label = self.df["label"][idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        X = torch.stack([input_ids, attention_mask], dim=-1)
        one_hot_label = torch.zeros(4, dtype=torch.float32)
        one_hot_label[label - 1] = 1.0

        return X, one_hot_label
