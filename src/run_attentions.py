"""
run_attentions.py 
------------
Compute and save attention flow matrices from a trained PLM model for either
classification or regression tasks.

This script uses the HIVSeqDataset class to ensure tokenization is consistent
with the training procedure.

Typical usage:
    # For a trained classifier replicate
    python src/run_attentions.py --task_type classification --input_csv data/input_VRC01_IC80.csv --model_path results/cv/PLM_classification_fold_2.pt --result_dir results/attention_maps/classification/fold_2 --pooling cls --head_hidden_dim 128 --dropout 0.3
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
from tqdm import tqdm
from transformers import AutoTokenizer

# Add local source directory to path for custom module imports
sys.path.insert(0, "src")
from model import PLMClassifier, PLMRegressor
from dataset import HIVSeqDataset
import utils as ut
from engine import Predictor

# ============ DEFAULTS ============
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
CLF_LABEL_NAME = "Label"
REG_LABEL_NAME = "pIC80"
INPUT_CSV = "data/input.csv"
RESULT_DIR = "results/attention_maps"
LOG_DIR = "logs/attn"
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute attention flow matrices from a trained PLM model.")
    parser.add_argument("--task_type", type=str, required=True, choices=["classification", "regression"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--head_hidden_dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--log_dir", type=str, default=LOG_DIR)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--load_weights", type=str, default="all", choices=["all", "head_only", "random"],
                        help="Specify whether to load 'all' weights or 'head_only' (keeps base PLM weights untouched), or 'random' all the weights.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()

def main():
    """Main execution routine for computing and saving attention flow matrices."""
    args = parse_args()
    g = ut.set_seed(args.seed)
    log_name = f"attn_flow_{args.task_type}"
    logger = ut.setup_logging(args.log_dir, log_name)

    # Logging setup
    logger.info("=================================================")
    logger.info("        Starting Attention Flow Computation      ")
    logger.info("=================================================")
    for key, value in vars(args).items():
        logger.info(f"  - {key}: {value}")
    logger.info(f"  - Using device: {DEVICE}")

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
    
    full_checkpoint = torch.load(args.model_path, map_location=DEVICE)
    if args.load_weights == "head_only":
        logger.info("Loading saved weights for the MLP head ONLY...")
        base_model_prefix = "plm"
        
        head_only_weights = {
            k: v for k, v in full_checkpoint.items() 
            if not k.startswith(base_model_prefix)
        }

        if not head_only_weights:
            logger.warning(f"No head weights found! Check if the base model prefix '{base_model_prefix}' is correct.")
        
        model.load_state_dict(head_only_weights, strict=False)
        logger.info("Head weights loaded successfully. Base PLM weights remain at original pre-trained values.")
    
    elif args.load_weights == "random":
        logger.info("Ignoring checkpoint. Randomizing ALL weights (base PLM + head)...")
        model.plm.init_weights()
        head = model.classifier if args.task_type == "classification" else model.regressor
        for layer in head:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        logger.info("All model weights have been completely randomized.")
    
    else:
        logger.info("Loading ALL saved weights (Base PLM + Head)...")
        model.load_state_dict(full_checkpoint, strict=True)
        logger.info("All weights loaded successfully.")
    
    #model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    predictor = Predictor(model=model, device=DEVICE)

    # --- 2. Prepare Output Directory ---
    os.makedirs(args.result_dir, exist_ok=True)

    # --- 3. Load Data and Instantiate Dataset ---
    df = pd.read_csv(args.input_csv)
    sequences = df["Sequence"].astype(str).tolist()
    max_actual_length = df["Sequence"].str.len().max()
    max_len = min(max_actual_length+2, args.max_len)
    logger.info(f"  - effective max_len: {max_len}")
    # if args.task_type == "regression":
    #     labels = df[REG_LABEL_NAME].astype(float).tolist()
    # else:
    #     labels = df[CLF_LABEL_NAME].astype(int).tolist()

    # Tokenize the sequences
    full_dataset = HIVSeqDataset(sequences, tokenizer, max_len)

    # --- 4. Process Each Sequence using the Dataset ---
    for i in tqdm(range(len(df)), desc="Processing sequences"):
        seq_no = df.iloc[i]["Seq_no"]

        try:
            # a. Get pre-tokenized data from the dataset
            tokens = full_dataset[i]
            # Add a batch dimension of 1, as the predictor expects a batch
            input_ids = tokens["input_ids"].unsqueeze(0)
            attention_mask = tokens["attention_mask"].unsqueeze(0)

            # b. Get model outputs using the Predictor
            _, attentions = predictor.predict(input_ids, attention_mask, return_attentions=True)

            # c. Filter attentions to exclude special tokens
            token_list_full = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
            aa_indices = [
                idx for idx, tok in enumerate(token_list_full)
                if tok not in tokenizer.all_special_tokens
            ]

            assert [token_list_full[i] for i in aa_indices] == list(df.iloc[i]["Sequence"])

            # prepending the <cls> token's index.
            valid_indices = []
            if token_list_full[0] == tokenizer.cls_token:
                valid_indices.append(0)
            valid_indices.extend(aa_indices)
            
            # d. Prepare and compute attention flow
            attn_tensor = torch.stack(attentions).squeeze(1) # Squeeze batch dim

            # save raw attentions
            raw_attn_filtered = attn_tensor[:, :, valid_indices, :][:, :, :, valid_indices]
            raw_filename = f"{seq_no}_attentions_raw.npy"
            raw_output_path = os.path.join(args.result_dir, raw_filename)
            np.save(raw_output_path, raw_attn_filtered.detach().cpu().numpy())

            attn_rollout = ut.compute_attention_rollout(attn_tensor)
            attn_rollout_filtered_rows = attn_rollout[valid_indices]
            attn_rollout_filtered = attn_rollout_filtered_rows[:, valid_indices]
            
            # e. Save the result with a robust filename
            filename = f"{seq_no}_attentions.npy"
            output_path = os.path.join(args.result_dir, filename)
            np.save(output_path, attn_rollout_filtered)

        except Exception as e:
            logger.error(f"Failed to process sequence {seq_no}: {e}", exc_info=True)

    logger.info("=================================================")
    logger.info("               Processing Complete               ")
    logger.info("=================================================")
    logger.info(f"Aggregrating attention matrices saved to: {args.result_dir}")

if __name__ == "__main__":
    main()
