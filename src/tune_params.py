"""
tune_params.py
--------------------------
Unified script for hyperparameter tuning of PLM models for classification and
regression tasks using Optuna.

Typical usage:
    # Tune a classifier
    python tune_params.py --task_type classification --n_trials 50

    # Tune a regressor
    python tune_params.py --task_type regression --n_trials 50
"""

# ============================
# Imports
# ============================
import os
import sys
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
from sklearn.utils.class_weight import compute_class_weight
import optuna

# Local imports
sys.path.insert(0, 'src')
from model import PLMClassifier, PLMRegressor
from dataset import HIVSeqDataset
import utils as ut
from engine import Trainer

# ============================
# Defaults
# ============================
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
CLF_LABEL_NAME = "Label"
REG_LABEL_NAME = "pIC80"
INPUT_CSV = "data/input.csv"
RESULTS_DIR = "results/tuning_optuna"
LOG_DIR = "logs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================

def setup_logging(log_dir, task_type):
    # ... (remains the same)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"optuna_{task_type}_{timestamp}.log")
    # ... (rest of function is the same)
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
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for PLM models with Optuna.")
    
    # --- Task Configuration ---
    parser.add_argument("--task_type", type=str, required=True,
                        choices=["classification", "regression"],
                        help="The type of task to perform.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (for classification).")
    
    # --- Optuna/CV Configuration ---
    parser.add_argument("--n_trials", type=int, default=50, help="Number of tuning trials for Optuna to run.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation.")
    
    # ... (other arguments remain the same)
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum epochs per fold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV, help="Path to input CSV file.")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR, help="Directory to save models and results.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Directory to save log files.")
    return parser.parse_args()


def objective(trial: optuna.Trial, args, full_dataset, stratify_labels, tokenizer, g) -> float:
    """
    Defines one Optuna trial, which runs a full K-Fold CV for a given set of hyperparameters.
    """
    # --- 1a. Suggest Hyperparameters ---
    params = {
        'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
        'unfreeze_layers': trial.suggest_int('unfreeze_layers', 5, 25),
        'batch_size': trial.suggest_categorical('batch_size', [4, 6, 8]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
    }

    # Add task-specific hyperparameters
    if args.task_type == "classification":
        params['head_hidden_dim'] = trial.suggest_categorical('head_hidden_dim', [0, 64, 128, 256, 512]) # 0 for linear head
        params['criterion'] = trial.suggest_categorical('criterion', ['CrossEntropy', 'WeightedCrossEntropy'])
    else: # regression
        params['criterion'] = trial.suggest_categorical('criterion', ['MSE', 'Huber'])

    logging.info(f"\n--- Starting Trial {trial.number} with params: {params} ---")

    # --- 1b. Run K-Fold Cross-Validation ---
    kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_losses = []
    stratify_labels_np = np.array(stratify_labels)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset, stratify_labels_np)):
        logging.info(f"--- Fold {fold + 1}/{args.n_splits} ---")

        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        train_loader = DataLoader(full_dataset, batch_size=params['batch_size'], sampler=train_sampler, generator=g)
        val_loader = DataLoader(full_dataset, batch_size=params['batch_size'], sampler=val_sampler)

        # --- Dynamically create model and criterion ---
        if args.task_type == "classification":
            model = PLMClassifier(model_name=MODEL_NAME, num_classes=args.num_classes,
                                  head_hidden_dim=params['head_hidden_dim'], dropout=params['dropout'])
            if params['criterion'] == 'WeightedCrossEntropy':
                train_labels = stratify_labels_np[train_ids]
                weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
                criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(DEVICE))
            else:
                criterion = nn.CrossEntropyLoss()
        else: # regression
            model = PLMRegressor(model_name=MODEL_NAME, dropout=params['dropout'])
            if params['criterion'] == 'Huber':
                criterion = nn.HuberLoss()
            else:
                criterion = nn.MSELoss()
        
        model = ut.freeze_all_but_last_n(model, params['unfreeze_layers'])
        model.to(DEVICE)
        
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=params['lr'])
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * args.epochs)

        # --- Instantiate and run the Trainer ---
        trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=DEVICE,
                          task_type=args.task_type, scheduler=scheduler, patience=args.patience)
        result = trainer.fit(train_loader=train_loader, val_loader=val_loader,
                             epochs=args.epochs, tokenizer=tokenizer)
        
        # Optuna will minimize the best validation loss found in this fold
        if result['best_model_state']:
            fold_losses.append(result['best_metrics']['loss'])
        else: # Handle case where training fails or makes no progress
            fold_losses.append(np.inf)

    # --- 1c. Return the final metric for Optuna to optimize ---
    mean_val_loss = np.mean(fold_losses)
    logging.info(f"--- Trial {trial.number} finished. Mean Val Loss: {mean_val_loss:.4f} ---")
    
    return mean_val_loss


def main():
    """Main execution routine for hyperparameter tuning with Optuna."""
    args = parse_args()
    logger = setup_logging(args.log_dir, args.task_type)

    logger.info("=================================================")
    logger.info(f" Starting PLM {args.task_type.capitalize()} Hyperparameter Tuning ")
    logger.info("=================================================")
    
    # --- Setup ---
    os.makedirs(args.results_dir, exist_ok=True)
    g = ut.set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Load Data ---
    df = pd.read_csv(args.input_csv)
    sequences = df["Sequence"].astype(str).tolist()
    
    if args.task_type == "regression":
        labels = df[REG_LABEL_NAME].astype(float).tolist()
    else: # classification
        labels = df[CLF_LABEL_NAME].astype(int).tolist()
    
    # Stratification is always done on the binned classification labels
    stratify_labels = df[CLF_LABEL_NAME].astype(int).tolist()
    full_dataset = HIVSeqDataset(sequences, labels, tokenizer, max_len=512)

    # --- Optuna Study ---
    storage_path = os.path.join(args.results_dir, f"optuna_study_{args.task_type}.db")
    storage_name = f"sqlite:///{storage_path}"
    study_name = f"plm-{args.task_type}-tuning-study"

    study = optuna.create_study(
        direction="minimize", # We always minimize validation loss
        storage=storage_name,
        study_name=study_name,
        load_if_exists=True
    )
    
    study.optimize(
        lambda trial: objective(trial, args, full_dataset, stratify_labels, tokenizer, g),
        n_trials=args.n_trials
    )

    # --- Report Best Results ---
    best_trial = study.best_trial
    logger.info("=" * 60)
    logger.info("          Hyperparameter Tuning Complete          ")
    logger.info("=" * 60)
    logger.info("Best trial found:")
    logger.info(f"  Value (Mean Val Loss): {best_trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save summary
    results_df = study.trials_dataframe()
    summary_path = os.path.join(args.results_dir, f"optuna_summary_{args.task_type}.csv")
    results_df.to_csv(summary_path, index=False)
    logger.info(f"Full tuning summary saved to {summary_path}")

if __name__ == "__main__":
    main()