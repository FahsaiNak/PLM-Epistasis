"""
attr.py
-------
Compute residue-level attribution scores from a trained PLM model using either
standard Integrated Gradients (IG), SmoothGrad-IG or InputXGradient

Typical usage:
    # Using standard IG
    python src/run_attributions.py --task_type classification --model_path results/final_classification_model_rep_1.pt --result_dir results/attribution_maps/classification/rep_1 --log_dir results/attribution_maps/classification/rep_1/log --method ig --target_class 0 --n_steps 1000

    # Using SmoothGrad-IG with 25 noisy samples
    python src/run_attributions.py --task_type classification --model_path results/final_classification_model_rep_1.pt --out_dir results/attribution_maps/classification/rep_1 --log_dir results/attribution_maps/classification/rep_1/log --method smoothgrad --target_class 0 --n_steps 250 --nt_samples 10

    # Using InputXGradient
    python src/run_attributions.py --task_type classification --model_path results/final_classification_model_rep_1.pt --out_dir results/attribution_maps/classification/rep_1 --log_dir results/attribution_maps/classification/rep_1/log --method gxi
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Local imports
sys.path.insert(0, "src")
from model import PLMClassifier, PLMRegressor
from dataset import HIVSeqDataset
import utils as ut

# ============ DEFAULTS ============
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
INPUT_CSV = "data/input.csv"
RESULT_DIR = "results/attribution_maps"
LOG_DIR = "logs/attr"
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute residue-level attributions for a trained PLM model.")
    
    # --- Attribution Configuration ---
    parser.add_argument("--method", type=str, default="ig", choices=["ig", "smoothgrad", "gxi"],
                        help="Attribution method to use.")
    parser.add_argument("--n_steps", type=int, default=100, help="Number of steps for Integrated Gradients.")
    parser.add_argument("--nt_samples", type=int, default=10, help="Number of noisy samples for SmoothGrad.")
    parser.add_argument("--stdevs", type=float, default=0.1, help="Standard deviation of noise for SmoothGrad.")
    parser.add_argument("--ig_threshold", type=float, default=0.05, help="Threshold for IG convergence delta.")
    parser.add_argument("--target_class", type=int, default=1, help="Target class index for classification attribution.")
    # --- Model Configuration ---
    parser.add_argument("--task_type", type=str, required=True, choices=["classification", "regression"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--head_hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    # --- Log Configuration ---
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--log_dir", type=str, default=LOG_DIR)
    return parser.parse_args()

def main():
    """Main execution routine for computing and saving attribution matrices."""
    args = parse_args()
    if args.method == "ig" or args.method == "smoothgrad":
        if args.task_type == "classification":
            log_name = f"attr_{args.method}_{args.target_class}"
    else:
        log_name = f"attr_{args.method}"
    logger = ut.setup_logging(args.log_dir, log_name)

    # Logging setup
    logger.info("=================================================")
    logger.info("         Starting Attribution Computation        ")
    logger.info("=================================================")
    for key, value in vars(args).items(): logger.info(f"  - {key}: {value}")
    logger.info(f"  - Using device: {DEVICE}")

    # Prepare Output Directory
    os.makedirs(args.result_dir, exist_ok=True)

    # --- 1. Load Model & Tokenizer ---
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if args.task_type == "regression":
        model = PLMRegressor(model_name=MODEL_NAME, dropout=args.dropout,
                             head_hidden_dim=args.head_hidden_dim, pooling_strategy=args.pooling)
    else: # classification
        model = PLMClassifier(model_name=MODEL_NAME, num_classes=args.num_classes,
                              head_hidden_dim=args.head_hidden_dim, dropout=args.dropout,
                              pooling_strategy=args.pooling)
    
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)

    # Load Data and Instantiate Dataset
    df = pd.read_csv(args.input_csv)
    sequences = df["Sequence"].astype(str).tolist()
    max_actual_length = df["Sequence"].str.len().max()
    max_len = min(max_actual_length+2, args.max_len)
    logger.info(f"  - effective max_len: {max_len}")

    full_dataset = HIVSeqDataset(sequences, tokenizer, max_len)
    full_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)

    # --- 2. Initialize Attribution Calculator ---
    calculator = ut.AttributionCalculator(model, args.task_type, target_class=args.target_class, pooling_strategy=args.pooling)

    # --- 3. Process Each Sequence ---
    warnings_count = 0
    for i, batch in enumerate(tqdm(full_loader, desc=f"Computing attributions with {args.method}")):
        seq_no = df.iloc[i]["Seq_no"]
        sequence = df.iloc[i]['Sequence']
        try:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # Call the method
            if args.method == 'smoothgrad':
                attribution, delta = calculator.compute_ig_with_smoothgrad(
                    sequence, input_ids, attention_mask,
                    n_steps=args.n_steps, nt_samples=args.nt_samples, stdevs=args.stdevs
                    )
            elif args.method == 'ig':
                attribution, delta = calculator.compute_ig(
                    sequence, input_ids, attention_mask, 
                    n_steps=args.n_steps
                    )
            elif args.method == 'gxi':
                args.target_class = None
                attribution = calculator.compute_gradient_x_input(
                    sequence, input_ids, attention_mask
                    )
                final_attribution = attribution
                delta = np.array(args.ig_threshold * 0.1)

            if (abs(delta) < args.ig_threshold).all():
                final_attribution = attribution
                save_output = True
            else:
                save_output = False
                mean_delta = np.mean(abs(delta))
                warnings_count += 1
            
            if save_output:
                filename = f"{seq_no}_attribution_{args.method}"
                if args.target_class is not None: filename += f"_class_{args.target_class}"
                filename += ".npy"
                output_path = os.path.join(args.result_dir, filename)
                np.save(output_path, final_attribution)
            else:
                logger.warning(f"High convergence delta ({mean_delta:.4f}) for {seq_no}. Skipping save.")

        
        except Exception as e:
            logger.error(f"Failed to process sequence {seq_no}: {e}", exc_info=True)
        
        torch.cuda.empty_cache()

    logger.info("=================================================")
    logger.info("               Processing Complete               ")
    logger.info("=================================================")
    if warnings_count > 0:
        logger.warning(f"Completed with {warnings_count} convergence warnings.")

if __name__ == "__main__":
    main()
