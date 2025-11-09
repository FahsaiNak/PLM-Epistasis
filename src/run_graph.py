"""
run_graph.py
-------
Builds a co-attribution/co-attention graph from saved model outputs
and computes residue centrality metrics.

This script loads pre-computed attribution and attention matrices for multiple
model replicates, averages them, and computes a single weighted graph
representing co-attention between residues, weighted by attribution scores.

Typical usage:
    python src/run_graph.py \
        --selected_models rep_5 rep_6 rep_7 rep_9 rep_10 \
        --attr_class class_1 \
        --weighted_by Source \
        --contribution Negative
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
import networkx as nx
from tqdm import tqdm

# Local imports
sys.path.insert(0, "src")
import utils as ut

# ============ DEFAULTS ============
INPUT_CSV = "data/input_info_VRC01_IC80.csv"
RESIDUE_INFO = "data/selected_residues_IC80.csv"
ATTR_DIR = "results/attribution_maps/classification/"
ATTN_DIR = "results/attention_maps/classification/full"
RESULT_DIR = "results/graphs"
LOG_DIR = "logs/graph"
# =================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build and analyze co-attention/co-attribution graphs.")
    
    # --- Path Configuration ---
    parser.add_argument("--attr_dir", type=str, default=ATTR_DIR,
                        help="Relative path to attribution maps directory.")
    parser.add_argument("--attn_dir", type=str, default=ATTN_DIR,
                        help="Relative path to attention maps directory.")
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR,
                        help="Relative path to save output graphs and metrics.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR,
                        help="Relative path to save log files.")

    # --- Model & Data Configuration ---
    parser.add_argument("--selected_models", type=str, nargs='+', required=True,
                        help="List of model replicate names (e.g., rep_1 rep_2).")
    parser.add_argument("--attr_class", type=str, required=True,
                        help="Attribution class to load (e.g., class_0, class_1).")
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV,
                        help="Path to the main input CSV file with sequence info.")
    parser.add_argument("--residue_info", type=str, default=RESIDUE_INFO,
                        help="Path to the CSV file with residue/resno info.")

    # --- Graph Configuration ---
    parser.add_argument("--weighted_by", type=str, choices=["Source", "Target", "None"], required=True,
                        help="How to weight attention: by Source, Target, or not at all (None).")
    parser.add_argument("--contribution", type=str, choices=["Negative", "Positive"], required=True,
                        help="Which attribution contribution to focus on.")

    return parser.parse_args()

def main():
    """Main execution routine for building and saving graphs."""
    args = parse_args()
    
    # Handle the 'None' string from argparse
    weighted_by_arg = args.weighted_by if args.weighted_by != "None" else None

    # --- Logging Setup ---
    log_name = f"graph_{args.attr_class}_{args.weighted_by}_{args.contribution}"
    logger = ut.setup_logging(args.log_dir, log_name)

    logger.info("=================================================")
    logger.info("          Starting Graph Construction            ")
    logger.info("=================================================")
    for key, value in vars(args).items(): logger.info(f"  - {key}: {value}")
    logger.info(f"  - weighted_by (processed): {weighted_by_arg}")

    # --- Prepare Output Directory ---
    os.makedirs(args.result_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {args.result_dir}")

    # --- Load Data ---
    logger.info("Loading data files...")
    data_csv_path = os.path.join(args.input_csv)
    resno_csv_path = os.path.join(args.residue_info)
    
    try:
        data_df = pd.read_csv(data_csv_path)
        resno_df = pd.read_csv(resno_csv_path)
    except FileNotFoundError as e:
        logger.error(f"Error: Data file not found. Make sure you are running from the project root. {e}")
        sys.exit(1)
        
    aligned_sequences = np.array([list(seq) for seq in data_df['Sequence']])
    resno_array = resno_df["ResLabel"].values
    logger.info(f"Loaded {len(data_df)} sequences and {len(resno_array)} residue labels.")

    N_MODEL = len(args.selected_models)
    N_SEQ, L_SEQ = aligned_sequences.shape
    logger.info(f"Processing {N_MODEL} models, {N_SEQ} sequences of length {L_SEQ}.")

    # --- Initialize Arrays ---
    attr_scalar_array = np.zeros((N_MODEL, N_SEQ, L_SEQ))
    attn_scalar_array = np.zeros((N_MODEL, N_SEQ, L_SEQ, L_SEQ))

    # --- Load Model Outputs ---
    logger.info(f"Loading attributions ({args.attr_class}) and attentions...")
    for n, model_name in enumerate(tqdm(args.selected_models, desc="Loading models")):
        for i, no in enumerate(data_df['Seq_no']):
            try:
                attr_path = os.path.join(args.attr_dir, model_name, f"{no}_attribution_ig_{args.attr_class}.npy")
                attr = np.load(attr_path)
                attr_scalar_array[n, i] = ut.magnitude_norm(attr)

                attn_path = os.path.join(args.attn_dir, model_name, f"{no}_attentions.npy")
                attn = np.load(attn_path)
                attn_arr = attn[1:, 1:] # removal of CLS tokens
                attn_scalar_array[n, i] = ut.min_max_norm(attn_arr)
            
            except FileNotFoundError as e:
                logger.warning(f"File not found for seq {no}, model {model_name}. Skipping. Details: {e}")
                # Ensure array remains 0s for this entry
                attr_scalar_array[n, i] = 0 
                attn_scalar_array[n, i] = 0
                continue
            except Exception as e:
                logger.error(f"Error processing seq {no}, model {model_name}: {e}", exc_info=True)
                continue

    # --- Process Arrays ---
    logger.info("Averaging arrays across models...")
    mean_attr_scalar_array = np.mean(attr_scalar_array, axis=0)
    mean_attn_scalar_array = np.mean(attn_scalar_array, axis=0)

    logger.info("Building type-aware arrays...")
    attr_array, attn_array, aa_idx_array = ut.build_typeaware_arrays(
        aligned_sequences, mean_attr_scalar_array, mean_attn_scalar_array
    )

    logger.info(f"Computing weighted attention (by={args.weighted_by}, contribution={args.contribution})...")
    weighted_attn = ut.compute_weighted_attention(
        attr_array, attn_array, weighted_by_arg, args.contribution
    )

    # --- Build and Save Graph ---
    logger.info("Building graph...")
    G = ut.build_graph(weighted_attn, resno_array)

    graphfile = os.path.join(args.result_dir, f"Graph_{args.attr_class}_{args.weighted_by}_{args.contribution}.gml")
    nx.write_gml(G, graphfile)
    logger.info(f"  Saved co-attention graph -> {graphfile}")

    # --- Compute and Save Centrality ---
    logger.info("Computing residue centrality...")
    df = ut.compute_residue_centrality(G)

    outfile = os.path.join(args.result_dir, f"Connectivity_{args.attr_class}_{args.weighted_by}_{args.contribution}.csv")
    df.to_csv(outfile, index=False)
    logger.info(f"  Saved residue connectivity metrics -> {outfile}")
    
    logger.info("=================================================")
    logger.info("               Processing Complete               ")
    logger.info("=================================================")

if __name__ == "__main__":
    main()
