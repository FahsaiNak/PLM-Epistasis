"""
tune_params.py
--------------------------
Script for hyperparameter tuning of the PLM regressor using Optuna
for efficient, intelligent search.

This script uses Optuna to explore a defined hyperparameter space. For each
trial, Optuna suggests a set of parameters, and the script evaluates them
using K-Fold Cross-Validation. The final output identifies the best-performing
configuration found during the search.

Typical usage:
    python tune_params.py --n_trials 50 --n_splits 5
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
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
import optuna

# Local imports
sys.path.insert(0, 'src')
from model import PLMRegressor
from dataset import HIVSeqDataset
import utils as ut
from engine import train_epoch, evaluate

# ============ DEFAULTS ============
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
LABEL_NAME = "pIC80"
INPUT_CSV = "data/input.csv"
RESULTS_DIR = "results/tuning_optuna"
LOG_DIR = "logs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================

def setup_logging(log_dir):
    # ... (This function remains the same)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"optuna_tuning_run_{timestamp}.log")
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
    # ... (This function remains the same)
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for a PLM Regressor with Optuna.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of tuning trials for Optuna to run.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum epochs per fold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV, help="Path to input CSV file.")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR, help="Directory to save models and results.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Directory to save log files.")
    return parser.parse_args()


# ==========================================================
# 1. DEFINE THE OBJECTIVE FUNCTION FOR OPTUNA
# ==========================================================
def objective(trial: optuna.Trial, args, full_dataset, class_labels, tokenizer, g) -> float:
    """
    This function defines one trial for Optuna. It will:
    1. Suggest a set of hyperparameters for the PLMRegressor.
    2. Run a full K-Fold Cross-Validation with these parameters.
    3. Return the average validation MSE, which Optuna will try to minimize.
    """
    # --- 1a. Suggest Hyperparameters ---
    params = {
        'lr': trial.suggest_float('lr', 1e-6, 1e-4, log=True),
        'unfreeze_layers': trial.suggest_int('unfreeze_layers', 5, 20),
        'batch_size': trial.suggest_categorical('batch_size', [4, 6, 8]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'num_heads': trial.suggest_categorical('num_heads', [1, 2, 4, 8]),
    }
    logging.info(f"\n--- Starting Trial {trial.number} with params: {params} ---")

    # --- 1b. Run K-Fold Cross-Validation ---
    kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset, class_labels)):
        logging.info(f"--- Fold {fold + 1}/{args.n_splits} ---")

        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        train_loader = DataLoader(full_dataset, batch_size=params['batch_size'], sampler=train_sampler, generator=g)
        val_loader = DataLoader(full_dataset, batch_size=params['batch_size'], sampler=val_sampler)

        model = PLMRegressor(
            model_name=MODEL_NAME,
            num_heads=params['num_heads'],
            dropout=params['dropout']
        )
        model = ut.freeze_all_but_last_n(model, params['unfreeze_layers'])
        model.to(DEVICE)

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=params['lr'])
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * args.epochs)
        criterion = nn.MSELoss()

        best_val_mse = np.inf
        patience_counter = 0
        for epoch in range(args.epochs):
            # The training and evaluation engine is model-agnostic and remains the same
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE, 1, 0.0, tokenizer)
            val_mse, _ = evaluate(model, val_loader, DEVICE)
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                logging.info(f"Early stopping in Fold {fold+1} at Epoch {epoch+1}")
                break
        
        fold_results.append(best_val_mse)

    # --- 1c. Return the final metric for Optuna to optimize ---
    mean_mse_for_trial = np.mean(fold_results)
    logging.info(f"--- Trial {trial.number} finished. Mean Val MSE: {mean_mse_for_trial:.4f} ---")
    
    return mean_mse_for_trial


def main():
    """Main execution routine for hyperparameter tuning with Optuna."""
    args = parse_args()
    logger = setup_logging(args.log_dir)

    logger.info("=================================================")
    logger.info("  Starting PLM Regressor Hyperparameter Tuning   ")
    logger.info("=================================================")
    
    # ... (The rest of the main function remains the same) ...
    os.makedirs(args.results_dir, exist_ok=True)
    g = ut.set_seed(args.seed)

    logger.info(f"Loading full dataset from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    sequences = df["Sequence"].astype(str).tolist()
    labels = df[LABEL_NAME].astype(float).tolist()
    class_labels = df["Label"].astype(int).tolist()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_dataset = HIVSeqDataset(sequences, labels, tokenizer, max_len=512)

    storage_path = os.path.join(args.results_dir, "optuna_study.db")
    storage_name = f"sqlite:///{storage_path}"
    study_name = "plm-regressor-tuning-study"

    logger.info(f"Using shared storage: {storage_name}")
    logger.info(f"Joining study: {study_name}")

    study = optuna.create_study(
        direction="minimize",
        storage=storage_name,
        study_name=study_name,
        load_if_exists=True
    )
    
    study.optimize(
        lambda trial: objective(trial, args, full_dataset, class_labels, tokenizer, g),
        n_trials=args.n_trials
    )

    best_trial = study.best_trial
    
    logger.info("=" * 60)
    logger.info("          Hyperparameter Tuning Complete          ")
    logger.info("=" * 60)
    logger.info(f"Number of finished trials in DB: {len(study.trials)}")
    logger.info("Best trial found across all workers:")
    logger.info(f"  Value (Mean Val MSE): {best_trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

    results_df = study.trials_dataframe()
    summary_path = os.path.join(args.results_dir, "optuna_tuning_summary.csv")
    results_df.to_csv(summary_path, index=False)
    logger.info(f"Full tuning summary saved to {summary_path}")

if __name__ == "__main__":
    main()