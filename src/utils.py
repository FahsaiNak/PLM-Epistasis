"""
utils.py
---------
Helper utilities for reproducibility, masking, and model analysis.
"""

import random
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from transformers import PreTrainedModel, AutoTokenizer

def set_seed(seed: int = 42) -> torch.Generator:
    """
    Set random seeds across all relevant libraries for reproducibility.

    This function sets the seed for Python's `random`, `numpy`, and `torch`
    (for both CPU and CUDA). It also returns a torch.Generator object,
    which can be passed to a DataLoader to ensure shuffled batches are
    reproducible.

    Parameters
    ----------
    seed : int, optional
        The integer value to use as the random seed. Default is 42.

    Returns
    -------
    torch.Generator
        A seeded generator object for use in PyTorch's DataLoader.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return g

def compute_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Pearson correlation coefficient between two arrays.

    Parameters
    ----------
    x : np.ndarray
        The first array of values.
    y : np.ndarray
        The second array of values.

    Returns
    -------
    float
        The Pearson correlation coefficient, a value between -1 and 1.
    """
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    return np.corrcoef(x, y)[0, 1]

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Mean Squared Error between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        The ground truth values.
    y_pred : np.ndarray
        The predicted values.

    Returns
    -------
    float
        The calculated Mean Squared Error.
    """
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def random_mask_tokens(input_ids: torch.Tensor, tokenizer: AutoTokenizer, mask_prob: float = 0.15) -> torch.Tensor:
    """
    Apply BERT-style dynamic masking to a batch of token IDs.

    For 15% of the tokens (excluding special tokens like [CLS], [SEP]), this
    function applies the following strategy:
    - 80% of the time: Replace the token with the [MASK] token.
    - 10% of the time: Replace the token with a random token from the vocabulary.
    - 10% of the time: Keep the original token (unchanged).

    Note: This function modifies the `input_ids` tensor in-place.

    Parameters
    ----------
    input_ids : torch.Tensor
        A batch of token IDs. Shape: (batch_size, seq_len).
    tokenizer : transformers.AutoTokenizer
        The tokenizer used for tokenization, to access special token IDs.
    mask_prob : float, optional
        The probability of a token being selected for masking. Default is 0.15.

    Returns
    -------
    torch.Tensor
        The `input_ids` tensor with masking applied.
    """
    if mask_prob <= 0:
        return input_ids

    device = input_ids.device
    mask_token_id = tokenizer.mask_token_id
    special_tokens = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}

    # 1. Determine which tokens to mask (ignoring special tokens)
    rand = torch.rand(input_ids.shape, device=device)
    mask_arr = (rand < mask_prob)

    is_special = torch.zeros_like(mask_arr)
    for t_id in special_tokens:
        is_special |= (input_ids == t_id)
    mask_arr &= ~is_special

    # 2. Get indices of tokens to be masked
    mask_idx = mask_arr.nonzero(as_tuple=True)
    if mask_idx[0].numel() == 0:
        return input_ids

    # 3. Apply the 80/10/10 masking strategy
    rand_sub = torch.rand(mask_idx[0].shape, device=device)

    # 80% of the time -> [MASK]
    mask_replace_mask = rand_sub < 0.8
    input_ids[mask_idx[0][mask_replace_mask], mask_idx[1][mask_replace_mask]] = mask_token_id

    # 10% of the time -> random token
    random_replace_mask = (rand_sub >= 0.8) & (rand_sub < 0.9)
    if random_replace_mask.any():
        num_random = random_replace_mask.sum()
        random_tokens = torch.randint(0, tokenizer.vocab_size, size=(num_random,), device=device)
        input_ids[mask_idx[0][random_replace_mask], mask_idx[1][random_replace_mask]] = random_tokens

    # The remaining 10% are left unchanged by default.
    return input_ids

def freeze_all_but_last_n(model: PreTrainedModel, n: Optional[int] = 10) -> PreTrainedModel:
    """
    Freeze model parameters to enable parameter-efficient fine-tuning.

    This function sets `requires_grad=False` for all model parameters except
    for the last `n` encoder layers and the final regression head. If `n` is
    None, all parameters are unfrozen.

    Parameters
    ----------
    model : torch.nn.Module
        The Transformer model to modify.
    n : int or None, optional
        The number of final encoder layers to keep unfrozen. If None, the entire
        model is made trainable. Default is 10.

    Returns
    -------
    torch.nn.Module
        The modified model with updated `requires_grad` flags.
    """
    if n is None:
        for p in model.parameters():
            p.requires_grad = True
        return model

    # Freeze all parameters initially
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze the last n encoder layers
    encoder_layers = model.plm.encoder.layer
    for layer in encoder_layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True

    # Unfreeze the regression head if it exists
    if hasattr(model, "regressor"):
        for p in model.regressor.parameters():
            p.requires_grad = True
    return model
