"""
train_cv.py
---------
Training script for fine-tuning PLM using K-Fold Cross-Validation.

This script handles the entire fine-tuning process, including data splitting
into K folds, model re-initialization for each fold, training, evaluation,
and early stopping based on validation performance.

Typical usage:
    python train.py --n_splits 5 --epochs 100 --lr 1e-5 --unfreeze_layers 12
"""

# ============================
# Imports
# ============================
import os
import sys
import math
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn, amp
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold, KFold

# Local imports
sys.path.insert(0, 'src')
from model import PLMRegressor
from dataset import HIVSeqDataset
import utils as ut
import engine as eg

# ============ DEFAULTS ============
MODEL_NAME = "facebook/esm2_t33_650M_UR50D" #"Rostlab/prot_bert"
LABEL_NAME = "pIC80"
INPUT_CSV = "data/input.csv"
RESULTS_DIR = "results"
LOG_DIR = "logs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================

def setup_logging(log_dir):
    """Configures the logging for the script."""
    # ... (same as your previous version)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_run_{timestamp}.log")
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

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune ProtBERT for regression with K-Fold Cross-Validation.")
    
    # --- K-Fold Configuration ---
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation.")

    # --- Paths and I/O ---
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV, help="Path to input CSV file.")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR, help="Directory to save models and results.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Directory to save log files.")

    # --- Training Hyperparameters ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=500, help="Max epochs per fold.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--num_pooling_head", type=int, default=4, help="Attention heads in pooling layer.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout value.")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--unfreeze_layers", type=int, default=5, help="Num layers to unfreeze.")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="MLM masking probability.")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation.")
    parser.add_argument("--warmup_steps", type=int, default=3, help="Scheduler warmup steps.")

    # --- Early Stopping ---
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--improvement_threshold", type=float, default=1e-5, help="Minimum improvement to reset patience.")

    return parser.parse_args()

def main():
    """Main execution routine for K-Fold Cross-Validation."""
    args = parse_args()
    logger = setup_logging(args.log_dir)
    
    logger.info("=================================================")
    logger.info("      Starting PLM Fine-Tuning with K-Fold CV    ")
    logger.info("=================================================")
    logger.info("Running with the following configuration:")
    for key, value in vars(args).items():
        logger.info(f"  - {key}: {value}")
    logger.info(f"  - Using device: {DEVICE}")

    # --- Setup ---
    os.makedirs(args.results_dir, exist_ok=True)
    g = ut.set_seed(args.seed)

    # --- Load Tokenizer ONCE ---
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Load Full Dataset ONCE ---
    logger.info(f"Loading full dataset from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    sequences = df["Sequence"].astype(str).tolist()
    labels = df[LABEL_NAME].astype(float).tolist()
    class_labels = df["Label"].astype(int).tolist()
    full_dataset = HIVSeqDataset(sequences, labels, tokenizer, args.max_len)
    
    # --- K-Fold Training Loop ---
    kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_results = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset, class_labels)):
        logger.info("-" * 50)
        logger.info(f"STARTING FOLD {fold + 1}/{args.n_splits}")
        logger.info("-" * 50)

        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        train_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=train_sampler, generator=g)
        val_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=val_sampler)

        # --- Re-initialize Model and Sync Embeddings INSIDE the loop ---
        logger.info("Re-initializing model for the new fold.")
        model = PLMRegressor(MODEL_NAME) #, pooling_type=args.pooling, dropout=args.dropout)
        # Train the last n layers
        model = ut.freeze_all_but_last_n(model, args.unfreeze_layers)
        model.to(DEVICE)

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        total_steps = math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
        criterion = nn.MSELoss()

        # --- Inner Training Loop for the current fold ---
        best_val_mse = np.inf
        patience_counter = 0
        fold_model_path = os.path.join(args.results_dir, f"PLM_regressor_fold_{fold+1}.pt")

        for epoch in range(args.epochs):
            train_loss = eg.train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE, args.accumulation_steps, args.mask_prob, tokenizer)
            val_mse, val_r = eg.evaluate(model, val_loader, DEVICE)
            
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss={train_loss:.4f} | Val MSE={val_mse:.4f} | Val PearsonR={val_r:.4f}")

            if val_mse < best_val_mse - args.improvement_threshold:
                # Improvement found, save the model and reset patience
                best_val_mse = val_mse
                best_val_r = val_r
                patience_counter = 0
                save_model = model.state_dict()
                logger.info(f"Improvement found! New best validation MSE: {best_val_mse:.4f}.")
            else:
                # No improvement, increment the patience counter
                patience_counter += 1
                logger.info(f"No improvement in validation MSE. Patience: {patience_counter}/{args.patience}")
                # If patience runs out, stop training
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered. Training finished.")
                    break
        
        # --- Save the best model  ---
        torch.save(save_model, fold_model_path)
        # --- Store the best result for this fold ---
        logger.info(f"Fold {fold+1} finished. Best Validation MSE: {best_val_mse:.4f}")
        # Reload the best model to get its final Pearson correlation
        model.load_state_dict(torch.load(fold_model_path))
        val_mse, val_r = eg.evaluate(model, val_loader, DEVICE)
        fold_results.append({'fold': fold + 1, 'best_val_mse': val_mse, 'best_val_pearsonr': val_r})

    # --- Aggregate and Log Final Results ---
    results_df = pd.DataFrame(fold_results)
    mean_mse = results_df['best_val_mse'].mean()
    std_mse = results_df['best_val_mse'].std()
    mean_r = results_df['best_val_pearsonr'].mean()
    std_r = results_df['best_val_pearsonr'].std()

    logger.info("=================================================")
    logger.info("         K-Fold Cross-Validation Summary         ")
    logger.info("=================================================")
    logger.info(f"Average Validation MSE: {mean_mse:.4f} +/- {std_mse:.4f}")
    logger.info(f"Average Validation Pearson's R: {mean_r:.4f} +/- {std_r:.4f}")
    logger.info("Individual fold results:")
    logger.info("\n" + results_df.to_string(index=False))
    
    # Save summary to CSV
    results_df.to_csv(os.path.join(args.results_dir, "kfold_summary.csv"), index=False)
    logger.info(f"K-Fold summary saved to {args.results_dir}/kfold_summary.csv")

if __name__ == "__main__":
    main()