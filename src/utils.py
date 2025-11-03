src/utils.py"""
utils.py
---------
Helper utilities for reproducibility, masking, and model analysis.
"""

import sys
import random
import os
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Union, Dict, Set, Literal
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoTokenizer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, mean_squared_error
from scipy.stats import pearsonr
from captum.attr import IntegratedGradients, NoiseTunnel, InputXGradient

def setup_logging(log_dir, task_type):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{task_type}_{timestamp}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    return logger

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

def get_special_tokens(tokenizer: AutoTokenizer) -> Set[int]:
    """Helper function to get all special token IDs from a tokenizer."""
    special_tokens = {
        tokenizer.cls_token_id,
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
        tokenizer.unk_token_id
    }
    # ESM uses <eos> while BERT uses [SEP]
    if hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None:
        special_tokens.add(tokenizer.sep_token_id)
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        special_tokens.add(tokenizer.eos_token_id)
        
    # Remove None if it's in the set
    return {tid for tid in special_tokens if tid is not None}

def random_mask_tokens(input_ids: torch.Tensor, tokenizer: AutoTokenizer, mask_prob: float = 0.15) -> torch.Tensor:
    """
    Apply ESM-style dynamic masking to a batch of token IDs.

    This function avoids modifying the input tensor in-place. For a random 15%
    of the non-special tokens, it applies the following strategy:
    - 80% of the time: Replace with the <mask> token.
    - 10% of the time: Replace with a random *amino acid* token.
    - 10% of the time: Keep the original token.

    Parameters
    ----------
    input_ids : torch.Tensor
        A batch of token IDs. Shape: (batch_size, seq_len).
    tokenizer : transformers.AutoTokenizer
        The tokenizer, used to access special token IDs and the amino acid vocabulary.
    mask_prob : float, optional
        The probability of a token being selected for masking. Default is 0.15.

    Returns
    -------
    torch.Tensor
        A new tensor with masking applied.
    """
    if mask_prob <= 0:
        return input_ids

    inputs = input_ids.clone()
    device = inputs.device
    
    special_token_ids = get_special_tokens(tokenizer)
    amino_acids = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']
    aa_token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(amino_acids), device=device)

    # 1. Determine which tokens are eligible for masking
    probability_matrix = torch.full(inputs.shape, mask_prob, device=device)
    is_special_token = torch.zeros_like(inputs, dtype=torch.bool)
    for token_id in special_token_ids:
        is_special_token |= (inputs == token_id)
    probability_matrix.masked_fill_(is_special_token, value=0.0)
    
    # Get the coordinates of all tokens selected for the 15% pool
    indices_masked = torch.bernoulli(probability_matrix).bool().nonzero(as_tuple=True)
    if indices_masked[0].numel() == 0:
        return inputs

    # 2. Decide which of the selected tokens get the <mask> token (80%)
    masked_subset_prob = torch.full(indices_masked[0].shape, 0.8, device=device)
    masked_subset_mask = torch.bernoulli(masked_subset_prob).bool()
    
    # Apply the <mask> token to the 80% subset
    mask_coords = (indices_masked[0][masked_subset_mask], indices_masked[1][masked_subset_mask])
    inputs[mask_coords] = tokenizer.mask_token_id

    # 3. Identify the remaining 20% of the pool
    remaining_mask = ~masked_subset_mask
    remaining_row_indices = indices_masked[0][remaining_mask]
    remaining_col_indices = indices_masked[1][remaining_mask]

    # 4. Of this remainder, replace half (10% of original) with a random amino acid
    if remaining_row_indices.numel() > 0:
        random_subset_prob = torch.full(remaining_row_indices.shape, 0.5, device=device)
        random_subset_mask = torch.bernoulli(random_subset_prob).bool()

        # Get the final coordinates for random replacement
        random_coords = (remaining_row_indices[random_subset_mask], remaining_col_indices[random_subset_mask])
        
        # Apply random amino acid tokens
        num_random = random_coords[0].numel()
        if num_random > 0:
            rand_aa_indices = torch.randint(0, len(aa_token_ids), (num_random,), device=device)
            random_tokens = aa_token_ids[rand_aa_indices]
            inputs[random_coords] = random_tokens
            
    # The final 10% are left unchanged by default
    return inputs

def evaluate_predictions(
    pred: List[Union[float, Dict[str, float]]],
    true: List[Union[int, float]],
    task_type: Literal["regression", "classification"]
) -> Dict[str, float]:
    """
    Computes final evaluation metrics from a list of predictions and true labels.

    This function is designed to work with the output of a Predictor class,
    handling both regression and classification tasks.

    Parameters
    ----------
    pred : List[Union[float, Dict[str, float]]]
        A list of model predictions. For regression, this is a list of floats.
        For classification, this is a list of dictionaries containing class
        probabilities (e.g., [{'Class_0': 0.1, 'Class_1': 0.9}, ...]).
    true : List[Union[int, float]]
        A list of the ground truth labels.
    task_type : str
        The type of task, either "regression" or "classification".

    Returns
    -------
    Dict[str, float]
        A dictionary containing the relevant evaluation metrics for the task.
    """
    if not pred:
        print("Warning: Prediction list is empty. Cannot compute metrics.")
        return {}

    metrics = {}
    true_labels_np = np.array(true)

    if task_type == "regression":
        preds_np = np.array(pred)
        
        metrics["mse"] = mean_squared_error(true_labels_np, preds_np)
        pearson_corr, _ = pearsonr(true_labels_np, preds_np)
        metrics["pearsonr"] = pearson_corr

    elif task_type == "classification":
        # For classification, we need to extract predicted labels and probabilities
        # from the list of dictionaries.

        # Get predicted class labels by finding the class with the highest probability
        pred_labels_np = np.array([np.argmax(list(p.values())) for p in pred])
        
        # Get the probability of the positive class (class 1) for AUC calculation
        # This assumes a binary classification task.
        positive_probs_np = np.array([p.get('Class_1', 0.0) for p in pred])

        metrics["accuracy"] = accuracy_score(true_labels_np, pred_labels_np)
        metrics["f1_score"] = f1_score(true_labels_np, pred_labels_np, average='weighted', zero_division=0)
        
        try:
            metrics["auc"] = roc_auc_score(true_labels_np, positive_probs_np)
        except ValueError:
            # This can happen if the true labels only contain one class
            metrics["auc"] = 0.5
            print("Warning: Could not compute AUC. Defaulting to 0.5. This may be due to only one class being present in the true labels.")

    else:
        raise ValueError(f"Unknown task_type: '{task_type}'. Must be 'regression' or 'classification'.")

    return metrics

def freeze_all_but_last_n(model: nn.Module, n: Optional[int] = 10) -> nn.Module:
    """
    Freezes model parameters to enable parameter-efficient fine-tuning.

    Sets `requires_grad=False` for all parameters except for the last `n`
    encoder layers and the final prediction head.

    Parameters
    ----------
    model : torch.nn.Module
        The Transformer model to modify.
    n : int or None
        The number of final encoder layers to keep unfrozen.
        - If n > 0, the last n layers are unfrozen.
        - If n == 0, the entire base model is frozen.
        - If n is None, the entire model is unfrozen.

    Returns
    -------
    torch.nn.Module
        The modified model.
    """
    if n is None:
        for param in model.parameters():
            param.requires_grad = True
        return model

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # --- Find the base model dynamically ---
    base_model = None
    common_names = ["bert", "esm", "plm"]
    for name in common_names:
        if hasattr(model, name):
            base_model = getattr(model, name)
            break

    if base_model is None:
        raise ValueError(f"Could not find a base model with common names {common_names} in the provided model.")

    # Unfreeze the last n encoder layers
    if n > 0:
        if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
            encoder_layers = base_model.encoder.layer
            for layer in encoder_layers[-n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            # Fallback for models that might not have the exact .encoder.layer structure
            # This part may need adjustment for different model types (e.g., T5)
            # For BERT-like models (ESM, ProtBERT), the above is sufficient.
            pass

    # --- Unfreeze any custom head(s) ---
    if hasattr(model, "regressor"):
        for param in model.regressor.parameters():
            param.requires_grad = True
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    return model

def compute_attention_rollout(attentions: torch.Tensor) -> torch.Tensor:
    # Initialize rollout with identity matrix
    rollout = torch.eye(attentions[0].size(-1)).to(attentions[0].device)

    # Multiply attention maps layer by layer
    for attention in attentions:
        attention_heads_fused = attention.mean(dim=0) # Average attention across heads
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device) # A + I
        attention_heads_fused /= attention_heads_fused.sum(dim=-1, keepdim=True) # Normalizing A
        rollout = torch.matmul(rollout, attention_heads_fused) # Multiplication

    return rollout.cpu().numpy()

class AttributionCalculator:
    """A self-contained class to compute attributions using Integrated Gradients."""
    def __init__(self, model, task_type, target_class=None, pooling_strategy="cls"):
        self.model = model
        self.task_type = task_type
        self.target_class = target_class
        self.pooling_strategy = pooling_strategy
        self.model.eval()

        if self.task_type == 'classification' and self.target_class is None:
            raise ValueError("`target_class` must be specified for classification attribution.")

    def _forward_func(self, input_embeds, attention_mask):
        # Captum requires a forward function that takes embeddings as input.
        outputs = self.model.plm(inputs_embeds=input_embeds, attention_mask=attention_mask)
        if self.pooling_strategy == 'cls':
            pooled_output = outputs.pooler_output
        elif self.pooling_strategy == 'mean':
            pooled_output = outputs.last_hidden_state.mean(dim=1)
        elif self.pooling_strategy == 'max':
            pooled_output = outputs.last_hidden_state.max(dim=1).values
        else:
            raise ValueError("pooling_strategy must be 'cls', 'mean' or 'max'")

        if self.task_type == "regression":
            return self.model.regressor(pooled_output)
        else: # classification
            logits = self.model.classifier(pooled_output)
            return logits[:, self.target_class]

    def compute_ig(self, sequence: str, input_ids: torch.Tensor, attention_mask: torch.Tensor, baseline_ids: Optional[torch.Tensor] = None, n_steps: int = 100, internal_batch_size: int = 10) -> tuple:
        """Computes Integrated Gradients for a sequence."""
        ig = IntegratedGradients(self._forward_func)
        if baseline_ids is None:
            baseline_ids = torch.zeros_like(input_ids)

        input_embeddings = self.model.plm.embeddings.word_embeddings(input_ids)
        baseline_embeddings = self.model.plm.embeddings.word_embeddings(baseline_ids)

        attributions, delta = ig.attribute(
            inputs=input_embeddings,
            baselines=baseline_embeddings,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            return_convergence_delta=True,
            internal_batch_size=internal_batch_size
        )

        attributions_sum = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()

        # Slice to remove special tokens and match original sequence length
        return attributions_sum[1 : len(sequence) + 1], delta.cpu().numpy()

    def compute_ig_with_smoothgrad(self, sequence: str, input_ids: torch.Tensor, attention_mask: torch.Tensor, baseline_ids: Optional[torch.Tensor] = None, n_steps: int = 100, nt_samples: int = 10, stdevs: float = 0.1, internal_batch_size: int = 10) -> tuple:
        """Computes Integrated Gradients with the SmoothGrad technique."""
        ig = IntegratedGradients(self._forward_func)
        nt = NoiseTunnel(ig)
        if baseline_ids is None:
            baseline_ids = torch.zeros_like(input_ids)

        input_embeddings = self.model.plm.embeddings(input_ids)
        baseline_embeddings = self.model.plm.embeddings(baseline_ids)

        attributions, delta = nt.attribute(
            inputs=input_embeddings,
            baselines=baseline_embeddings,
            nt_type='smoothgrad',
            nt_samples=nt_samples,
            stdevs=stdevs,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            return_convergence_delta=True,
            internal_batch_size=internal_batch_size
        )

        attributions_sum = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        # Slice to remove special tokens and match original sequence length
        return attributions_sum[1 : len(sequence) + 1], delta.cpu().numpy()

    def compute_gradient_x_input(self, sequence: str, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        """Computes Gradient x Input attributions for a sequence."""
        gxi = InputXGradient(self._forward_func)
        input_embeddings = self.model.plm.embeddings.word_embeddings(input_ids)

        attributions = gxi.attribute(
            inputs=input_embeddings,
            additional_forward_args=(attention_mask,)
        )

        attributions_sum = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        
        # Slice to remove special tokens and match original sequence length
        return attributions_sum[1 : len(sequence) + 1]
