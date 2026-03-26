import torch
from torch.nn.functional import cosine_similarity
from gensim.models import KeyedVectors

from src.const import LOGGER

DEFAULT_WORD_LIST = [
    # Business Terms
    "finance", "economy", "investment", "market", "stocks", "banking", "entrepreneur", "startup", "corporate", "trade",
    
    # Science Terms
    "psychology", "physics", "chemistry", "scientist", "professor", "biology", "ecology", "molecular", "quantum", "astronomy",
    
    # Sports Terms
    "football", "basketball", "tennis", "athlete", "coach", "tournament", "league", "championship", "olympics", "score",
    
    # World Terms
    "conflict", "battle", "treaty", "climate", "terrorism", "refugee", "iran", "israel", "war", "syria"
    
]

def keyword_masking(input_ids: torch.Tensor, kv: dict, word_list: list[str] | None = None) -> torch.Tensor:
    if word_list is None:
        word_list = DEFAULT_WORD_LIST
    
    # Pre-compute masked token IDs for faster lookup
    masked_ids = {kv[word] for word in word_list if word in kv}
    mask_token = kv["[MASK]"]
    
    # Vectorized masking operation
    masked_input = input_ids.clone()
    mask_tensor = torch.tensor(mask_token, dtype=input_ids.dtype, device=input_ids.device)
    
    # Create boolean mask for all positions
    is_masked = torch.isin(input_ids.flatten(), torch.tensor(list(masked_ids), dtype=input_ids.dtype, device=input_ids.device))
    masked_input.flatten()[is_masked] = mask_tensor
    
    return masked_input.view_as(input_ids)
    