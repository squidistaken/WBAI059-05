import torch
from src.utils.data import TorchDataset
from src.const import LOGGER, DEBUG
from typing import Optional, Dict

DEFAULT_WORD_LIST = [
    # Business Terms
    "finance",
    "economy",
    "investment",
    "market",
    "stocks",
    "banking",
    "entrepreneur",
    "startup",
    "corporate",
    "trade",
    # Science Terms
    "psychology",
    "physics",
    "chemistry",
    "scientist",
    "professor",
    "biology",
    "ecology",
    "molecular",
    "quantum",
    "astronomy",
    # Sports Terms
    "football",
    "basketball",
    "tennis",
    "athlete",
    "coach",
    "tournament",
    "league",
    "championship",
    "olympics",
    "score",
    # World Terms
    "conflict",
    "battle",
    "treaty",
    "climate",
    "terrorism",
    "refugee",
    "iran",
    "israel",
    "war",
    "syria",
]


def keyword_masking(
    input_ids: torch.Tensor,
    kv: Dict[str, int],
    word_list: Optional[list[str]] = None,
) -> torch.Tensor:
    """Mask specific keywords in the input IDs by replacing them with the
    [MASK] token ID.

    Args:
        input_ids (torch.Tensor): The input IDs to be masked.
        kv (Dict[str, int]): The tokenizer's vocabulary mapping words to token
                             IDs.
        word_list (Optional[list[str]], optional): The list of words to be
                                                   masked. Defaults to None.

    Returns:
        torch.Tensor: The input IDs with specified keywords masked.
    """
    if word_list is None:
        word_list = DEFAULT_WORD_LIST

    # Pre-compute masked token IDs for faster lookup
    masked_ids = {kv[word] for word in word_list if word in kv}
    mask_token = kv["[MASK]"]

    # Vectorized masking operation
    masked_input = input_ids.clone()
    mask_tensor = torch.tensor(
        mask_token, dtype=input_ids.dtype, device=input_ids.device
    )

    # Create boolean mask for all positions
    is_masked = torch.isin(
        input_ids.flatten(),
        torch.tensor(
            list(masked_ids), dtype=input_ids.dtype, device=input_ids.device
        ),
    )
    masked_input.flatten()[is_masked] = mask_tensor

    return masked_input.view_as(input_ids)


def split_length_buckets(
    input_ids: torch.Tensor, labels: torch.Tensor, bucket_size: int = 50
) -> Dict[str, TorchDataset]:
    """Split the dataset into buckets based on the length of the input
    sequences.

    Args:
        input_ids (torch.Tensor): The input IDs of the dataset.
        labels (torch.Tensor): The corresponding labels of the dataset.
        bucket_size (int, optional): The size of each bucket. Defaults to 50.

    Returns:
        Dict[str, TorchDataset]: A dictionary mapping bucket names to their
                                 respective datasets.
    """
    # We assume padding token ID is 0
    lengths = (input_ids != 0).sum(dim=1)[:, 0]
    buckets = {}

    if DEBUG:
        LOGGER.debug(
            f"Input IDs shape: {input_ids.shape}, Lengths: {lengths}, Bucket Size: {bucket_size}"
        )

    for start in range(0, input_ids.size(1), bucket_size):
        end = start + bucket_size
        bucket_mask = (lengths > start) & (lengths <= end)

        if DEBUG:
            LOGGER.debug(
                f"Bucket {start}-{end}: Mask Sum: {bucket_mask.sum()}, Mask Shape: {bucket_mask.shape}"
            )

        buckets[f"{start}-{end}"] = TorchDataset(
            input_ids[bucket_mask], labels[bucket_mask]
        )

    return buckets
