"""
engine.py
---------
This module provides the core helper functions for training and evaluating
PyTorch models. It includes a training loop with mixed precision and gradient
accumulation, an evaluation loop for calculating metrics, and an inference
function for making predictions on single sequences.
"""

import sys
from typing import Tuple, Union, Dict, Literal, Optional
import numpy as np
import torch
from torch import nn, amp
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from transformers import AutoTokenizer, PreTrainedModel
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, mean_squared_error
from scipy.stats import pearsonr

# Local imports
sys.path.insert(0, 'src')
import utils as ut
from model import PLMClassifier, PLMRegressor

class Evaluator:
    """
    A flexible evaluation class for regression and classification tasks.

    This class accumulates predictions and labels batch by batch and computes
    a dictionary of relevant metrics upon request using task-specific methods.

    Parameters
    ----------
    task_type : str
        The type of task, either "regression" or "classification".
    """
    def __init__(self, task_type: Literal["regression", "classification"]):
        self.task_type = task_type
        self.reset()

    def reset(self):
        """Clears the stored predictions and labels for a new evaluation round."""
        self._outputs = []
        self._true = []

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Updates the evaluator's state with a new batch of outputs and labels.

        Parameters
        ----------
        outputs : torch.Tensor
            The raw output (logits or predictions) from the model.
        labels : torch.Tensor
            The ground truth labels.
        """
        self._outputs.append(outputs.cpu())
        self._true.append(labels.cpu())

    def compute_regression(self) -> Dict[str, float]:
        """Computes and returns regression-specific metrics (MSE, Pearson's r)."""
        final_preds = torch.cat(self._outputs).numpy()
        final_true = torch.cat(self._true).numpy()
        
        metrics = {}
        metrics["mse"] = mean_squared_error(final_true, final_preds)
        pearson_corr, _ = pearsonr(final_true, final_preds)
        metrics["pearsonr"] = pearson_corr
        return metrics

    def compute_classification(self) -> Dict[str, float]:
        """Computes and returns classification-specific metrics (accuracy, F1, AUC)."""
        final_logits = torch.cat(self._outputs)
        final_true = torch.cat(self._true).numpy()
        
        final_probs = torch.softmax(final_logits, dim=1).numpy()
        final_preds = np.argmax(final_probs, axis=1)

        metrics = {}
        metrics["accuracy"] = accuracy_score(final_true, final_preds)
        metrics["f1_score"] = f1_score(final_true, final_preds, average='weighted', zero_division=0)
        
        positive_probs = final_probs[:, 1] if final_probs.shape[1] == 2 else final_probs
        try:
            metrics["auc"] = roc_auc_score(final_true, positive_probs, multi_class='ovr')
        except ValueError:
            metrics["auc"] = 0.5
        return metrics

    def compute(self) -> Dict[str, float]:
        """
        Computes the final evaluation metrics by dispatching to the correct method.

        Returns
        -------
        Dict[str, float]
            A dictionary of relevant evaluation metrics for the specified task.
        """
        if self.task_type == "regression":
            return self.compute_regression()
        elif self.task_type == "classification":
            return self.compute_classification()
        else:
            raise ValueError(f"Unknown task_type: '{self.task_type}'")

class Trainer:
    """
    A class to encapsulate the training and validation loop for a PyTorch model.

    This class manages the model, optimizer, loss function, and device, and
    provides a high-level `fit` method to run the entire training process,
    including early stopping and model saving.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    optimizer : Optimizer
        The optimizer for updating model weights.
    criterion : nn.Module
        The loss function.
    device : torch.device
        The device to perform computation on.
    task_type : str
        The task type, either "regression" or "classification".
    scheduler : _LRScheduler, optional
        The learning rate scheduler. Default is None.
    patience : int, optional
        Patience for early stopping. Default is 10.
    accumulation_steps : int, optional
        Gradient accumulation steps. Default is 1.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        task_type: Literal["regression", "classification"],
        *,
        scheduler: Optional[_LRScheduler] = None,
        patience: int = 10,
        accumulation_steps: int = 1
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.task_type = task_type
        self.scheduler = scheduler
        self.patience = patience
        self.accumulation_steps = accumulation_steps

        # Initialize state for training
        self.best_metrics = None
        self.best_metric = np.inf  # We monitor validation loss, so lower is better
        self.patience_counter = 0
        self.best_model_state = None
        self.evaluator = Evaluator(task_type=self.task_type)

    def _train_epoch(self, loader: DataLoader, tokenizer: AutoTokenizer, mask_prob: float) -> float:
        """
        Runs a single epoch of training. This is the logic from your original function.
        """
        self.model.train()
        scaler = amp.GradScaler(enabled=(self.device.type == "cuda"))
        total_loss = 0.0

        self.optimizer.zero_grad()
        for step, batch in enumerate(tqdm(loader, desc="Training Epoch")):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            if self.task_type == "regression": labels = labels.float()
            else: labels = labels.long()

            if mask_prob > 0:
                input_ids = ut.random_mask_tokens(input_ids, tokenizer, mask_prob)

            with amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                preds, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(preds, labels) / self.accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * self.accumulation_steps

            if (step + 1) % self.accumulation_steps == 0 or (step + 1) == len(loader):
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

        return total_loss / len(loader)

    def _validate_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Runs a single epoch of validation."""
        self.evaluator.reset()
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                if self.task_type == "regression": labels = labels.float()
                else: labels = labels.long()

                outputs, _ = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                self.evaluator.update(outputs, labels)
        
        metrics = self.evaluator.compute()
        metrics['loss'] = total_loss / len(loader)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, tokenizer: AutoTokenizer, mask_prob: float = 0.15) -> Dict[str, any]:
        """
        Runs the main training and validation loop for a specified number of epochs.
        """
        for epoch in range(epochs):
            logging.info(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            # --- Training ---
            train_loss = self._train_epoch(train_loader, tokenizer, mask_prob)
            logging.info(f"Train -> loss={train_loss:.4f}")

            # --- Validation ---
            val_metrics = self._validate_epoch(val_loader)
            log_str = "Valid -> " + " | ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
            logging.info(log_str)

            # --- Early Stopping Logic ---
            if val_metrics['loss'] < self.best_metric:
                self.best_metric = val_metrics['loss']
                self.best_metrics = val_metrics
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict()
                logging.info(f"Improvement! New best validation loss: {self.best_metric:.4f}.")
            else:
                self.patience_counter += 1
                logging.info(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
                if self.patience_counter >= self.patience:
                    logging.info("Early stopping triggered.")
                    break
        
        return {
            "best_model_state": self.best_model_state,
            "best_metrics": self.best_metrics
        }

class Predictor:
    """
    A class to handle batch inference for PLM models from tokenized inputs.
    """
    def __init__(self, model: Union[PLMRegressor, PLMClassifier], device: torch.device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_attentions: bool = False
    ) -> Union[List[Union[float, Dict[str, float]]], Tuple[List[Union[float, Dict[str, float]]], tuple]]:
        """
        Makes predictions on a batch of pre-tokenized protein sequences.

        Parameters
        ----------
        input_ids : torch.Tensor
            A tensor of token IDs for the batch. Shape: (batch_size, seq_len).
        attention_mask : torch.Tensor
            The attention mask corresponding to the input_ids. Shape: (batch_size, seq_len).
        return_attentions : bool, optional
            If True, returns a tuple of (predictions, attentions). Default is False.

        Returns
        -------
        Union[List, Tuple]
            - If `return_attentions` is False: A list of predictions. Each item is a
              float (for regression) or a dict of probabilities (for classification).
            - If `return_attentions` is True: A tuple `(predictions_list, attentions)`.
        """
        # 1. Move tokenized inputs to the correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # 2. Perform inference
        with torch.no_grad():
            outputs, attentions = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # 3. Post-process the batch output
        if isinstance(self.model, PLMRegressor):
            # Convert the tensor of predictions to a list of floats
            predictions = outputs.cpu().numpy().flatten().tolist()
        
        elif isinstance(self.model, PLMClassifier):
            # Convert the tensor of logits to a list of probability dictionaries
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            predictions = [
                {f"Class_{i}": prob for i, prob in enumerate(p_row)}
                for p_row in probabilities
            ]
        
        else:
            raise TypeError("Model type not supported by this predictor.")
            
        # 4. Return the appropriate value
        if return_attentions:
            return predictions, attentions
        else:
            return predictions
