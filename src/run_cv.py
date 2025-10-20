"""
run_cv.py
---------
Unified training script for fine-tuning a PLM for classification or regression
using K-Fold Cross-Validation.

Select the task and loss function via command-line arguments.

Typical usage:
    # Classification
    python src/run_cv.py --task_type classification --criterion CrossEntropy --batch_size 6 --lr 5e-06 --unfreeze_layers 25 --dropout 0.1 --head_hidden_dim 64

    # Regression
    python src/run_cv.py --task_type regression --criterion Huber --batch_size 6 --lr 2e-05 --unfreeze_layers 25 --dropout 0.1
"""

# ============================
# Imports
# ============================
import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Local imports
sys.path.insert(0, 'src')
from model import PLMClassifier, PLMRegressor # Import both models
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
RESULTS_DIR = "results"
LOG_DIR = "logs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================

def parse_args():
    """Parse command-line arguments for classification and regression."""
    parser = argparse.ArgumentParser(description="Fine-tune a PLM with K-Fold CV.")

    # --- Task Configuration ---
    parser.add_argument("--task_type", type=str, required=True,
                        choices=["classification", "regression"],
                        help="The type of task to perform.")

    # --- Model Configuration ---
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (for classification).")
    parser.add_argument("--criterion", type=str, required=True, choices=["CrossEntropy", "WeightedCrossEntropy", "MSE", "Huber"])
    parser.add_argument("--head_hidden_dim", type=int, default=0, help="Hidden dim for MLP head. Set to 0 for a linear head. (for classification)")

    # -- Hyperparameters ---
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV, help="Path to input CSV file.")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR, help="Directory to save models and results.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Directory to save log files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per fold.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout value for the prediction head.")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--unfreeze_layers", type=int, default=5, help="Num layers to unfreeze.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")

    return parser.parse_args()

def main():
    """Main execution routine for K-Fold Cross-Validation."""
    args = parse_args()
    logger = ut.setup_logging(args.log_dir, args.task_type)

    logger.info(f"=================================================")
    logger.info(f"   Starting PLM Fine-Tuning for {args.task_type.upper()}   ")
    logger.info(f"=================================================")
    logger.info("Running with the following configuration:")
    for key, value in vars(args).items():
        logger.info(f"  - {key}: {value}")
    logger.info(f"  - Using device: {DEVICE}")

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

    # Use the class labels for stratification in both tasks
    stratify_labels = df[CLF_LABEL_NAME].astype(int).tolist()
    full_dataset = HIVSeqDataset(sequences, labels, tokenizer, args.max_len)

    # --- K-Fold Cross-Validation ---
    kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_results = []
    stratify_labels_np = np.array(stratify_labels)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset, stratify_labels_np)):
        logger.info("-" * 50)
        logger.info(f"STARTING FOLD {fold + 1}/{args.n_splits}")

        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        train_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=train_sampler, generator=g)
        val_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=val_sampler)

        # --- Initialize Model based on Task ---
        if args.task_type == "regression":
            model = PLMRegressor(model_name=MODEL_NAME, dropout=args.dropout)
        else: # classification
            model = PLMClassifier(model_name=MODEL_NAME, num_classes=args.num_classes, head_hidden_dim=args.head_hidden_dim, dropout=args.dropout)
        
        model = ut.freeze_all_but_last_n(model, args.unfreeze_layers)
        model.to(DEVICE)

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * args.epochs)

        if args.criterion == 'Huber': criterion = nn.HuberLoss()
        elif args.criterion == 'MSE': criterion = nn.MSELoss()
        elif args.criterion == 'CrossEntropy': criterion = nn.CrossEntropyLoss()
        else: # WeightedCrossEntropy
            class_labels = df[CLF_LABEL_NAME].astype(int).values
            weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(DEVICE))

            # --- Instantiate and run the Trainer ---
            trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=DEVICE,
                            task_type=args.task_type, scheduler=scheduler, patience=args.patience)
            result = trainer.fit(train_loader=train_loader, val_loader=val_loader,
                                epochs=args.epochs, tokenizer=tokenizer)

        # --- Save results from the fold ---
        if result["best_model_state"]:
            save_model_name = f"PLM_{args.task_type}_fold_{fold+1}.pt"
            fold_model_path = os.path.join(args.results_dir, save_model_name)
            torch.save(result["best_model_state"], fold_model_path)
            fold_results.append({'fold': fold + 1, **result["best_metrics"]})
            logger.info(f"Fold {fold+1} complete. Best validation loss: {result['best_metrics']['loss']:.4f}")

    # --- Aggregate and Log Final Results ---
    if fold_results:
        results_df = pd.DataFrame(fold_results)
        logger.info("=" * 60)
        logger.info("       K-Fold Cross-Validation Summary        ")
        logger.info("=" * 60)

        if args.task_type == "classification":
            mean_f1, std_f1 = results_df['f1_score'].mean(), results_df['f1_score'].std()
            mean_auc, std_auc = results_df['auc'].mean(), results_df['auc'].std()
            logger.info(f"Average Validation F1-Score: {mean_f1:.4f} +/- {std_f1:.4f}")
            logger.info(f"Average Validation AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        else: # regression
            mean_mse, std_mse = results_df['mse'].mean(), results_df['mse'].std()
            mean_r, std_r = results_df['pearsonr'].mean(), results_df['pearsonr'].std()
            logger.info(f"Average Validation MSE: {mean_mse:.4f} +/- {std_mse:.4f}")
            logger.info(f"Average Validation Pearson's R: {mean_r:.4f} +/- {std_r:.4f}")
        
        logger.info("Individual fold results:")
        logger.info("\n" + results_df.to_string(index=False))
        results_df.to_csv(os.path.join(args.results_dir, f"kfold_summary_{args.task_type}.csv"), index=False)
    else:
        logger.info("Cross-validation finished, but no results were recorded.")

if __name__ == "__main__":
    main()