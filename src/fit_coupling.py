import numpy as np
import pandas as pd
import os
import pickle
import warnings
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import KFold
from sklearn.utils import resample
from tqdm import tqdm
warnings.filterwarnings("ignore")

data_dir = "/pl/active/rdi_data/fahsai/PLM"

AA_LIST = list("ACDEFGHIKLMNPQRSTVWYO-")
N_AA = len(AA_LIST)

n_splits = 5
cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=42)
l1_ratio = 0.001
n_bootstraps = 1000
n_permutations = 500
min_N = 3

resno_df = pd.read_csv(os.path.join("data/selected_residues_IC80.csv"))
resno_array = resno_df["ResLabel"].values

with open(data_dir+'/results/full/aligned_sequences.pkl', 'rb') as c:
    aligned_sequences = pickle.load(c)

with open(data_dir+'/results/full/mean_attr_score_array.pkl', 'rb') as c:
    mean_attr_score_array = pickle.load(c)

with open(data_dir+'/results/full/com_map.pkl', 'rb') as c:
    mean_residue_com_map = pickle.load(c)

N_total, L = aligned_sequences.shape

epistatic_map = {}
raw_matrices = {} 

print(f"Initiating global coupling-weight extraction across {L} structural positions...")

# =====================================================================
# FEATURE EXTRACTION
# =====================================================================
# Outer loop with tqdm for tracking overall cluster job progress
for target_idx in tqdm(range(L), desc="Processing Trimer Positions"):
    target_reslabel = resno_array[target_idx]
    
    # Only evaluate amino acid states that actually physically exist in the alignment
    unique_aas = np.unique(aligned_sequences[:, target_idx])
    
    for target_aa in unique_aas:
        seq_mask = (aligned_sequences[:, target_idx] == target_aa).reshape(-1)
        selected_sequences = aligned_sequences[seq_mask]
        N_state = selected_sequences.shape[0]
        
        # Minimum Support Filter: Skip extremely rare states
        if N_state < min_N:
            continue

        # Data Setup
        E = mean_attr_score_array[seq_mask][:, target_idx]
        h = np.median(E)
        Y = (E - h).reshape(-1).copy()
        
        T = mean_residue_com_map[seq_mask][:, target_idx].reshape(N_state, L)
        O = (selected_sequences[:, :, np.newaxis] == np.array(AA_LIST)).astype(int)
        X = T[:, :, np.newaxis] * O
        X_flat = X.reshape(N_state, L * N_AA).copy()

        # --- A. Fit the Ground Truth Model ---
        if N_state <= n_splits:
            model_actual = ElasticNet(l1_ratio=l1_ratio)
            model_actual.fit(X_flat, Y)
            best_alpha = model_actual.alpha
        else:
            model_actual = ElasticNetCV(cv=cv_strategy, l1_ratio=l1_ratio)
            model_actual.fit(X_flat, Y)
            best_alpha = model_actual.alpha_

        actual_weights = model_actual.coef_.copy()

        # --- B. Bootstrapping (Stability Selection) ---
        selection_counts = np.zeros(X_flat.shape[1]) 
        subsample_size = max(2, int(0.8 * N_state))
        
        for b in range(n_bootstraps):
            X_sub, Y_sub = resample(X_flat, Y, n_samples=subsample_size, replace=False)
            model_boot = ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio, max_iter=2000)
            model_boot.fit(X_sub, Y_sub)
            selection_counts[model_boot.coef_ != 0] += 1

        boot_conf_flat = selection_counts / n_bootstraps  

        # --- C. Permutation Testing (Empirical P-Values) ---
        exceed_counts = np.zeros_like(actual_weights)

        for p in range(n_permutations):
            Y_shuffled = np.random.permutation(Y)
            model_perm = ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio, max_iter=2000)
            model_perm.fit(X_flat, Y_shuffled)
            exceed_counts += (np.abs(model_perm.coef_) >= np.abs(actual_weights)).astype(int)

        p_values_flat = (exceed_counts + 1) / (n_permutations + 1)
        p_values_flat[actual_weights == 0.0] = 1.0 # Insignificant if pruned

        # --- D. Sequence Support Metric ---
        support_flat = np.sum(O, axis=0).flatten()

        # =====================================================================
        # SPARSE DICTIONARY COMPRESSION
        # =====================================================================
        W_mat = actual_weights.reshape(L, N_AA)
        C_mat = boot_conf_flat.reshape(L, N_AA)
        P_mat = p_values_flat.reshape(L, N_AA)
        S_mat = support_flat.reshape(L, N_AA)
        
        # Save dense data temporarily for the save_raw flag
        raw_matrices[(target_reslabel, target_aa)] = {
            'Weights': W_mat, 'Boot_Conf': C_mat, 'P_Values': P_mat, 'Support': S_mat
        }

        # Isolate true structural dependencies
        non_zero_src_idx, non_zero_aa_idx = np.nonzero(W_mat)
        
        for src_idx, aa_idx in zip(non_zero_src_idx, non_zero_aa_idx):
            src_reslabel = resno_array[src_idx]
            src_aa = AA_LIST[aa_idx]
            
            key = (target_reslabel, target_aa, src_reslabel, src_aa)
            
            metrics_tuple = (
                W_mat[src_idx, aa_idx],  
                C_mat[src_idx, aa_idx],  
                P_mat[src_idx, aa_idx],  
                S_mat[src_idx, aa_idx]   
            )
            epistatic_map[key] = metrics_tuple

print(f"\nGlobal mapping complete! Extracted {len(epistatic_map)} absolute dependencies.")

# =====================================================================
# DATA SERIALIZATION
# =====================================================================
filepath = data_dir+"/results/coupling/epistatic_map.pkl"
payload = {'epistatic_map': epistatic_map, 'raw_matrices': raw_matrices}
with open(filepath, 'wb') as f:
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Successfully saved to {filepath}")
