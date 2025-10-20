"""
run_full_train.py
-----------------
Final training script to fine-tune a PLM on the entire dataset, followed by
an evaluation on that same training data to assess goodness of fit.

Typical usage:
    # Train and evaluate a final classifier with the best hyperparameters
    python src/run_full_train.py --task_type classification --num_replicates 3 --criterion CrossEntropy --batch_size 6 --lr 5e-06 --unfreeze_layers 25 --dropout 0.1 --head_hidden_dim 64
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
REG_LABEL_NAME = "pIC80"
INPUT_CSV = "data/input.csv"
RESULTS_DIR = "results"
LOG_DIR = "logs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train a final PLM on the full dataset.")
    parser.add_argument("--task_type", type=str, required=True, choices=["classification", "regression"])
    parser.add_argument("--criterion", type=str, required=True, choices=["CrossEntropy", "WeightedCrossEntropy", "MSE", "Huber"])
    parser.add_argument("--num_replicates", type=int, default=1, help="Number of training replicates to run with different seeds.")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--head_hidden_dim", type=int, default=64)
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--log_dir", type=str, default=LOG_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--unfreeze_layers", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    return parser.parse_args()

def main():
    """Main execution routine for training and evaluating final models."""
    args = parse_args()
    logger = ut.setup_logging(args.log_dir, args.task_type)

    logger.info("=================================================")
    logger.info(f"   Starting Final PLM Training ({args.num_replicates} Replicates)   ")
    logger.info("=================================================")
    logger.info("Running with the following configuration:")
    for key, value in vars(args).items():
        logger.info(f" - {key}: {value}")
    logger.info(f" - Using device: {DEVICE}")

    # --- Load Data Once ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = pd.read_csv(args.input_csv)
    sequences = df["Sequence"].astype(str).tolist()

    if args.task_type == "regression":
        labels = df[REG_LABEL_NAME].astype(float).tolist()
    else:
        labels = df[CLF_LABEL_NAME].astype(int).tolist()

    # This list will store the final metrics from each replicate run
    replicate_results = []

    # --- Main Replicate Loop ---
    for rep in range(args.num_replicates):
        current_seed = args.seed * rep
        logger.info(f"\n" + "="*60)
        logger.info(f"  Starting Replicate {rep + 1}/{args.num_replicates} with Seed {current_seed}  ")
        logger.info("="*60)

        # --- Setup for the current replicate ---
        os.makedirs(args.results_dir, exist_ok=True)
        g = ut.set_seed(current_seed)
        
        full_dataset = HIVSeqDataset(sequences, labels, tokenizer, args.max_len)
        full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, generator=g)

        # --- Initialize Model ---
        if args.task_type == "regression":
            model = PLMRegressor(model_name=MODEL_NAME, dropout=args.dropout)
        else:
            model = PLMClassifier(model_name=MODEL_NAME, num_classes=args.num_classes,
                                  head_hidden_dim=args.head_hidden_dim, dropout=args.dropout)
        model = ut.freeze_all_but_last_n(model, args.unfreeze_layers)
        model.to(DEVICE)

        # --- Initialize Optimizer, Scheduler, and Criterion ---
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(full_loader) * args.epochs)
        
        # ... (criterion selection logic remains the same) ...
        if args.criterion == 'Huber': criterion = nn.HuberLoss()
        elif args.criterion == 'MSE': criterion = nn.MSELoss()
        elif args.criterion == 'CrossEntropy': criterion = nn.CrossEntropyLoss()
        else: # WeightedCrossEntropy
            class_labels = df[CLF_LABEL_NAME].astype(int).values
            weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(DEVICE))

        # --- Training ---
        trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=DEVICE, 
                          task_type=args.task_type, scheduler=scheduler, patience=args.patience)
        result = trainer.fit(train_loader=full_loader, val_loader=full_loader, epochs=args.epochs, tokenizer=tokenizer)

        # --- Save the Model for the current replicate ---
        if result["best_model_state"]:
            logger.info(f"Replicate {rep+1} training completed. Best validation loss: {result['best_metrics']['loss']:.4f}")
            final_model_name = f"final_{args.task_type}_model_rep_{rep+1}.pt"
            final_model_path = os.path.join(args.results_dir, final_model_name)
            torch.save(result["best_model_state"], final_model_path)
            logger.info(f"Replicate {rep+1} model saved to: {final_model_path}")

            # --- Evaluation for the current replicate ---
            inference_model = model
            inference_model.load_state_dict(torch.load(final_model_path, map_location=DEVICE))
            predictor = Predictor(model=inference_model, device=DEVICE)

            eval_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
            all_predictions = []
            for batch in tqdm(eval_loader, desc=f"Predicting (Rep {rep+1})"):
                batch_predictions = predictor.predict(batch["input_ids"], batch["attention_mask"])
                all_predictions.extend(batch_predictions)
            
            eval_metrics = ut.evaluate_predictions(pred=all_predictions, true=labels, task_type=args.task_type)
            replicate_results.append(eval_metrics)

            summary_df = pd.DataFrame({"Sequence": sequences, "True": labels, "Prediction": all_predictions})
            summary_path = os.path.join(args.results_dir, f"training_results_{args.task_type}_rep_{rep+1}.csv")
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Replicate {rep+1} summary saved to: {summary_path}")
        else:
            logger.warning(f"Replicate {rep+1} did not improve and no model was saved.")

    # --- Aggregate and Log Final Summary ---
    if replicate_results:
        results_df = pd.DataFrame(replicate_results)
        logger.info("\n" + "="*60)
        logger.info(f"  Aggregated Performance Summary ({args.num_replicates} Replicates)  ")
        logger.info("="*60)
        
        for metric in results_df.columns:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            logger.info(f"  - Mean {metric.upper()}: {mean_val:.4f} +/- {std_val:.4f}")

        summary_path = os.path.join(args.results_dir, f"summary_{args.task_type}.csv")
        results_df.to_csv(summary_path, index_label="replicate")
        logger.info(f"\nAggregated summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
