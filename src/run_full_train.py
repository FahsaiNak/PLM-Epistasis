"""
run_full_train.py
-----------------
Final training script to fine-tune a PLM on the entire dataset using specified
hyperparameters, followed by evaluation on the training data.

Use this script AFTER hyperparameter tuning to train your final model.

Typical usage:
    # Train a final classifier with specific hyperparameters
    python src/run_full_train.py \
        --task_type classification --criterion CrossEntropy \
        --dropout 0.3 --head_hidden_dim 128 \
        --pooling cls --unfreeze_layers 9 \
        --batch_size 6 --lr 1.68e-5 \
        --epochs 50 \
        --input_csv data/input_VRC01_IC80.csv
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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# Local imports
sys.path.insert(0, 'src')
from model import PLMClassifier, PLMRegressor
from dataset import HIVSeqDataset
import utils as ut
from engine import Trainer, Predictor

# ============================
# Defaults
# ============================
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
CLF_LABEL_NAME = "Label"
REG_LABEL_NAME = "Value"
INPUT_CSV = "data/input.csv"
RESULT_DIR = "results/full"
LOG_DIR = "logs/full"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================

def parse_args():
    """Parse command-line arguments for the final training run."""
    parser = argparse.ArgumentParser(description="Train a final PLM on the full dataset.")

    # --- Task & Model Configuration ---
    parser.add_argument("--task_type", type=str, required=True, choices=["classification", "regression"], help="Task type.")
    parser.add_argument("--criterion", type=str, required=True, choices=["CrossEntropy", "WeightedCrossEntropy", "MSE", "Huber"], help="Loss function.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (for classification).")
    parser.add_argument("--head_hidden_dim", type=int, default=0, help="Hidden dim for MLP head (0 for linear).")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean", "maxmean"], help="Pooling strategy.")

    # --- Paths and I/O ---
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV, help="Path to the full dataset CSV.")
    parser.add_argument("--val_csv", type=str, default=INPUT_CSV, help="Path to the validation dataset CSV.")
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR, help="Directory to save the final model and results.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Directory to save log files.")
    parser.add_argument("--output_model_name", type=str, default=None, help="Optional name for the saved model file (e.g., final_model.pt).")

    # --- Final Training Hyperparameters ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, required=True, help="Total number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--unfreeze_layers", type=int, default=None, help="Number of PLM layers to unfreeze (None for all).")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")

    return parser.parse_args()

def main():
    """Main execution routine for training a final model and evaluating it."""
    args = parse_args()
    logger = ut.setup_logging(args.log_dir, f"full_train_{args.task_type}")

    logger.info("=================================================")
    logger.info(f"   Starting Final PLM Training for {args.task_type.upper()}   ")
    logger.info("=================================================")
    logger.info("Running with the following final configuration:")
    for key, value in vars(args).items():
        logger.info(f"  - {key}: {value}")
    logger.info(f"  - Using device: {DEVICE}")

    # --- Setup ---
    os.makedirs(args.result_dir, exist_ok=True)
    g = ut.set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Load Full Dataset ---
    logger.info(f"Loading full dataset from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    sequences = df["Sequence"].astype(str).tolist()
    max_actual_length = df["Sequence"].str.len().max()
    max_len = min(max_actual_length+2, args.max_len)
    logger.info(f"  - effective max_len: {max_len}")

    val_df = pd.read_csv(args.val_csv)
    val_sequences = val_df["Sequence"].astype(str).tolist()

    if args.task_type == "regression":
        labels = df[REG_LABEL_NAME].astype(float).tolist()
        val_labels = val_df[REG_LABEL_NAME].astype(float).tolist()
    else: # classification
        labels = df[CLF_LABEL_NAME].astype(int).tolist()
        val_labels = val_df[CLF_LABEL_NAME].astype(int).tolist()

    full_dataset = HIVSeqDataset(sequences, tokenizer, max_len, labels=labels)
    val_dataset = HIVSeqDataset(val_sequences, tokenizer, max_len, labels=val_labels)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Initialize Model ---
    if args.task_type == "regression":
        model = PLMRegressor(model_name=MODEL_NAME, dropout=args.dropout,
                             head_hidden_dim=args.head_hidden_dim, pooling_strategy=args.pooling)
    else: # classification
        model = PLMClassifier(model_name=MODEL_NAME, num_classes=args.num_classes,
                              head_hidden_dim=args.head_hidden_dim, dropout=args.dropout,
                              pooling_strategy=args.pooling)

    model = ut.freeze_all_but_last_n(model, args.unfreeze_layers)
    model.to(DEVICE)

    # --- Initialize Optimizer, Scheduler, and Criterion ---
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(full_loader) * args.epochs)

    if args.criterion == 'Huber': criterion = nn.HuberLoss()
    elif args.criterion == 'MSE': criterion = nn.MSELoss()
    elif args.criterion == 'CrossEntropy': criterion = nn.CrossEntropyLoss()
    else: # WeightedCrossEntropy
        class_labels_for_weighting = df[CLF_LABEL_NAME].astype(int).values
        weights = compute_class_weight('balanced', classes=np.unique(class_labels_for_weighting), y=class_labels_for_weighting)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(DEVICE))

    # --- Full Training Loop (No Validation/Early Stopping) ---
    logger.info("Starting final training loop on all data...")
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=DEVICE, task_type=args.task_type, scheduler=scheduler, patience=args.patience)
    result = trainer.fit(train_loader=full_loader, val_loader=val_loader, epochs=args.epochs, tokenizer=tokenizer)

    # --- Save the Final Model ---
    if args.output_model_name:
        final_model_name = args.output_model_name
        if not final_model_name.endswith('.pt'):
            final_model_name += '.pt'
    else:
        final_model_name = f"PLM_{args.task_type}_model.pt"

    final_model_path = os.path.join(args.result_dir, final_model_name)
    torch.save(result["best_model_state"], final_model_path)
    logger.info(f"Training complete. Best validation loss: {result['best_metrics']['loss']:.4f}")
    logger.info(f"Final model saved to: {final_model_path}")

    # --- Final Evaluation on Training Set (Goodness of Fit) ---
    logger.info("=================================================")
    logger.info("     Evaluating Final Model on Training Set      ")
    logger.info("=================================================")

    # Instantiate Predictor with the trained model
    if args.task_type == "regression":
        model = PLMRegressor(model_name=MODEL_NAME, dropout=args.dropout, head_hidden_dim=args.head_hidden_dim, pooling_strategy=args.pooling)
    else: # classification
        model = PLMClassifier(model_name=MODEL_NAME, num_classes=args.num_classes, head_hidden_dim=args.head_hidden_dim, dropout=args.dropout, pooling_strategy=args.pooling)
    model.load_state_dict(torch.load(final_model_path, map_location=DEVICE))
    predictor = Predictor(model=model, device=DEVICE)

    # Run predictions in batches (no shuffling for consistent output order)
    eval_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
    all_predictions = []
    for batch in tqdm(eval_loader, desc="Predicting on training data"):
        batch_predictions = predictor.predict(batch["input_ids"], batch["attention_mask"])
        all_predictions.extend(batch_predictions)

    # Compute and log metrics
    eval_metrics = ut.evaluate_predictions(
        pred=all_predictions,
        true=labels,
        task_type=args.task_type
    )
    logger.info("--- Training Set Performance (Goodness of Fit) ---")
    for metric, value in eval_metrics.items():
        logger.info(f"  - Final Training {metric.upper()}: {value:.4f}")

    # Save summary CSV
    summary_df = pd.DataFrame({
        "Sequence": sequences,
        "TrueLabel": labels,
        "Prediction": all_predictions
    })
    summary_path = os.path.join(args.result_dir, "training_summary_results.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Training summary with predictions saved to: {summary_path}")

if __name__ == "__main__":
    main()
