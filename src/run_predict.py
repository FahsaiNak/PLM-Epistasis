"""
run_predict.py
--------------
Script for making predictions on new data using a fine-tuned PLM model.

This script loads a trained classifier or regressor, processes a CSV file of
protein sequences, and saves the model's predictions to a new output file.

Typical usage:
    # For a trained classifier
    python src/run_predict.py --task_type classification  --input_csv data/input_VRC01_IC80.csv --model_path results/full/PLM_classification_model.pt --result_dir results/predictions/classification --output_file train.csv --pooling cls --head_hidden_dim 128 --dropout 0.3
    python src/run_predict.py --task_type classification  --input_csv data/input_VRC01_IC80.csv --model_path results/cv/PLM_classification_fold_2.pt --result_dir results/predictions/classification --output_file train_fold_2.csv --pooling cls --head_hidden_dim 128 --dropout 0.3
    python src/run_predict.py --task_type classification  --input_csv data/test_from_exp.csv --model_path results/cv/PLM_classification_fold_2.pt --result_dir results/predictions/classification --output_file test_exp_fold_2.csv --pooling cls --head_hidden_dim 128 --dropout 0.3

"""

# ============================
# Imports
# ============================
import os
import sys
import argparse
import logging
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# Local imports
sys.path.insert(0, 'src')
from model import PLMClassifier, PLMRegressor
from dataset import HIVSeqDataset
import utils as ut
from engine import Predictor

# ============================
# Defaults
# ============================
# This should be the same base model used for training
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
INPUT_CSV = "data/input.csv"
RESULT_DIR = "results"
LOG_DIR = "logs/predict"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================

def parse_args():
    """Parse command-line arguments for prediction."""
    parser = argparse.ArgumentParser(description="Make predictions with a trained PLM model.")

    # --- Task & Model Configuration ---
    parser.add_argument("--task_type", type=str, required=True, choices=["classification", "regression"],
                        help="The type of task the model was trained for.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (for classification model).")
    parser.add_argument("--head_hidden_dim", type=int, default=0,
                        help="Hidden dim for MLP head used during training. Set to 0 for a linear head.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value used during training.")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean", "max"])

    # --- Paths and I/O ---
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV, help="Path to the CSV file with sequences to predict.")
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR, help="Directory to save the prediction results.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Directory to save log files.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional: Specify a custom name for the output predictions CSV file.")
    
    # --- Data Handling ---
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length for tokenization.")
    
    return parser.parse_args()

def main():
    """Main execution routine for making predictions."""
    args = parse_args()
    logger = ut.setup_logging(args.log_dir, args.task_type)

    logger.info("=================================================")
    logger.info(f"      Starting Prediction for {args.task_type.upper()}      ")
    logger.info("=================================================")
    for key, value in vars(args).items():
        logger.info(f"  - {key}: {value}")
    logger.info(f"  - Using device: {DEVICE}")

    # --- 1. Load Model & Tokenizer ---
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Dynamically instantiate the correct model architecture
    if args.task_type == "regression":
        model = PLMRegressor(MODEL_NAME, head_hidden_dim=args.head_hidden_dim, dropout=args.dropout, pooling_strategy=args.pooling)
    else: # classification
        model = PLMClassifier(MODEL_NAME, num_classes=args.num_classes, head_hidden_dim=args.head_hidden_dim, dropout=args.dropout, pooling_strategy=args.pooling)

    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    logger.info("Model loaded successfully.")

    # --- 2. Instantiate Predictor ---
    predictor = Predictor(model=model, device=DEVICE)

    # --- 3. Load and Prepare Data ---
    logger.info(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    sequences = df["Sequence"].astype(str).tolist()
    max_actual_length = df["Sequence"].str.len().max()
    max_len = min(max_actual_length+2, args.max_len)
    logger.info(f"  - effective max_len: {max_len}")
    
    predict_dataset = HIVSeqDataset(sequences, tokenizer, max_len)
    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False)

    # --- 4. Run Prediction Loop ---
    all_predictions = []
    logger.info(f"Running inference on {len(df)} sequences...")
    for batch in tqdm(predict_loader, desc="Predicting"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # The predictor is designed to handle batches
        batch_predictions = predictor.predict(input_ids, attention_mask)
        all_predictions.extend(batch_predictions)

    # --- 5. Save Results ---
    # Add the predictions as a new column to the original dataframe
    df['Prediction'] = all_predictions
    
    # Create a descriptive output filename
    os.makedirs(args.result_dir, exist_ok=True)
    if args.output_file:
        output_filename = args.output_file
    else:
        output_filename = f"predictions_{args.task_type}.csv"
    if not output_filename.endswith('.csv'):
        output_filename += '.csv'
    output_path = os.path.join(args.result_dir, output_filename)
    df.to_csv(output_path, index=False)
    
    logger.info("=================================================")
    logger.info("                Prediction Complete              ")
    logger.info("=================================================")
    logger.info(f"Predictions for {len(df)} sequences saved to: {output_path}")

if __name__ == "__main__":
    main()
