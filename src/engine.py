"""
engine.py
---------
This module provides the core helper functions for training and evaluating
PyTorch models. It includes a training loop with mixed precision and gradient
accumulation, an evaluation loop for calculating metrics, and an inference
function for making predictions on single sequences.
"""

import sys
from typing import Tuple, Union

import torch
from torch import nn, amp
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

# Local imports
sys.path.insert(0, 'src')
import utils as ut

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    accumulation_steps: int,
    mask_prob: float,
    tokenizer: AutoTokenizer
) -> float:
    """
    Trains the model for one epoch.

    This function handles a single pass of training, incorporating key
    optimization techniques such as automatic mixed precision (AMP) for
    faster training on compatible GPUs and gradient accumulation to simulate
    a larger effective batch size.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    loader : DataLoader
        The DataLoader providing training batches.
    optimizer : Optimizer
        The optimizer for updating model weights.
    scheduler : _LRScheduler
        The learning rate scheduler.
    criterion : nn.Module
        The loss function (e.g., nn.MSELoss).
    device : torch.device
        The device to perform computation on ('cuda' or 'cpu').
    accumulation_steps : int
        The number of steps to accumulate gradients before updating weights.
    mask_prob : float
        The probability of applying random masking to input tokens.
    tokenizer : AutoTokenizer
        The tokenizer, required for the random masking utility.

    Returns
    -------
    float
        The average training loss for the epoch, normalized per sample.
    """
    model.train()
    # Automatic mixed precision scaler, enabled only for CUDA devices
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))
    total_loss = 0.0

    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Training Epoch")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Apply MLM-style dynamic masking if specified
        if mask_prob > 0:
            input_ids = ut.random_mask_tokens(input_ids, tokenizer, mask_prob)

        # Autocast context manager for mixed precision
        with amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            # Assuming the model returns (predictions, attentions)
            preds, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            # Normalize the loss by the number of accumulation steps
            loss = criterion(preds, labels) / accumulation_steps

        # Scale the loss and perform backward pass
        scaler.scale(loss).backward()
        total_loss += loss.item() * accumulation_steps

        # Perform optimizer step after accumulating gradients
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            # Unscale gradients before clipping to prevent distortion
            scaler.unscale_(optimizer)
            # Clip gradients to a max norm of 1.0 to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    # Return the average loss over the total number of samples
    return total_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluates the model on a given dataset.

    Parameters
    ----------
    model : nn.Module
        The model to be evaluated.
    loader : DataLoader
        The DataLoader providing evaluation batches.
    device : torch.device
        The device to perform computation on.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the Mean Squared Error (MSE) and the Pearson
        correlation coefficient (r).
    """
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            with amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                preds, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            all_preds.append(preds.cpu())
            all_true.append(labels.cpu())

    final_preds = torch.cat(all_preds).numpy()
    final_true = torch.cat(all_true).numpy()

    mse = ut.compute_mse(final_true, final_preds)
    r = ut.compute_pearsonr(final_true, final_preds)
    return mse, r


def predict(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a prediction and attention maps from already tokenized inputs.

    Parameters
    ----------
    model : nn.Module
        The trained model.
    input_ids : torch.Tensor
        A tensor of token IDs.
    attention_mask : torch.Tensor
        A tensor of attention masks.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - pred (torch.Tensor): The model's prediction tensor.
        - attention (torch.Tensor or tuple): Attention weights from the model.
    """
    model.eval()
    with torch.no_grad():
        pred, attention = model(input_ids=input_ids, attention_mask=attention_mask)
    return pred, attention
