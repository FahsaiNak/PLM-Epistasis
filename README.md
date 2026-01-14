# Decoding Viral Escape using Co-Attention Attribution from Protein Language Models

We fine-tune a protein language model (**ESM-2**) to predict HIV-1 Env (gp160) neutralization sensitivity to the broadly neutralizing antibody **VRC01**, and use **co-attention** and **attribution (Integrated Gradients)** analyses to interpret the molecular basis of viral escape.

---

## Overview

| Stage | Description |
|:------|:-------------|
| **Model training** | Fine-tune ESM on labeled gp140 sequences (sensitive vs. resistant to VRC01) |
| **Model evaluation** | Evaluate model performance and save per-sequence predictions. |
| **Attention extraction** | Compute [attention rollout](https://doi.org/10.48550/arXiv.2005.00928) to identify residue–residue relationships. |
| **Attribution analysis** | Use [Integrated Gradients (Captum)](https://captum.ai/docs/extension/integrated_gradients) to estimate residue-level contributions to class predictions. |
| **Residue communication** | Construct directional communication networks integrating attention and attribution. |
| **Interpretation** | Identify epitope features, escape pathways, and communication hubs |

---

## Workflow Summary

This project decodes the molecular basis of **HIV-1 escape from the broadly neutralizing antibody VRC01** using **ESM-based protein language models** and **co-attention attribution** analysis.

The full workflow consists of seven major stages:

---

### **1. Data Preparation**
**Goal:** Build a curated dataset linking HIV-1 gp140 sequences to VRC01 neutralization outcomes.

**Steps:**
- Collect gp140 (Env ectodomain) amino acid sequences.
- Gather experimental IC80 values (neutralization potency) from literature or neutralization databases.
- Label each sequence as:
  - `1` → **VRC01-sensitive**
  - `0` → **VRC01-escape**
- Save formatted data in `data/input_VRC01_IC80.csv`.

**Output:**  
`input_VRC01_IC80.csv` — input dataset for model training and evaluation.

---

### **2. Model Fine-Tuning**
**Goal:** Train a transformer-based classifier to distinguish VRC01-sensitive vs. escape sequences.

**Method:**
- **Backbone:** `facebook/esm2_t33_650M_UR50D`
- **Classifier:** 2-layer MLP (`hidden_dim = 128`)
- **Pooling:** `[CLS]` token
- **Unfrozen layers:** Top 9 of ESM
- **Loss:** CrossEntropy
- **Optimizer:** AdamW (`lr = 1.68e-5`, `weight_decay = 0.01`)
- **Dropout:** 0.3

**Procedure:**
- Optimize hyperparameters using [Optuna](https://optuna.org/)
- Train 10 replicate models (different random seeds for robustness).
- Save trained weights after 50 epochs or if validation loss does not improve for 10 consecutive epochs.

**Run:**
```bash
python src/run_full_train.py --task_type classification --input_csv data/input_VRC01_IC80.csv
```

**Output:**  
`results/full/PLM_classification_model_rep_{i}.pt` — fine-tuned model checkpoints.

---

### **3. Prediction and Evaluation**
**Goal:** Generate predictions and evaluate model consistency.

**Steps:**
- Use each trained model to predict VRC01 sensitivity for all sequences.
- Record predicted probabilities and class labels.
- Compare predictions across replicates to assess reproducibility.

**Run:**
```bash
python src/run_predict.py --task_type classification --input_csv data/input_VRC01_IC80.csv
```

**Output:**  
`results/predictions/classification/train_rep_{i}.csv` — per-sequence predictions.

---

### **4. Attention Extraction**
**Goal:** Identify residue–residue dependencies captured by the model.

**Method:**
- Extract attention matrices from ESM’s transformer layers.
- Aggregrate attention across layers ([attention rollout](https://doi.org/10.48550/arXiv.2005.00928)), and average over heads.
- Compare patterns between sensitive and escape groups to detect co-evolving residues.

**Run:**
```bash
python src/run_attentions.py --task_type classification --input_csv data/input_VRC01_IC80.csv
```

**Output:**  
`results/attention_maps/classification/rep_{i}/` — co-attention matrices (`.npy`).

---

### **5. Attribution Analysis (Integrated Gradients)**
**Goal:** Quantify residue-level contributions to model predictions.

**Method:**
- Apply **[Integrated Gradients (Captum)](https://captum.ai/docs/extension/integrated_gradients)** to compute per-residue attributions.
- Run for both target classes:
  - `target_class = 1` → VRC01-sensitive
  - `target_class = 0` → VRC01-escape
- Aggregate attribution results across replicates for stability analysis.

**Run:**
```bash
python src/run_attributions.py --task_type classification --target_class 0 --input_csv data/input_VRC01_IC80.csv
python src/run_attributions.py --task_type classification --target_class 1 --input_csv data/input_VRC01_IC80.csv
```

**Output:**  
`results/attribution_maps/classification/rep_{i}/` — residue-level importance scores.

---

### **6. Residue Communication and Goodness of Communication**

**Goal:** Resolve how residue-level effects propagate through sequence context.

**Method:**
- Construct two directional communication matrices by weighting residue–residue attention with [CLS]-based residue importance.
1. Influential (Outgoing): how effectively a residue broadcasts signals
2. Vulnerability (Incoming): how strongly a residue receives contextual signals
- Integrate attribution scores to quantify functional signal propagation.
- Derive Goodness of Communication scores (net, positive, and negative) to identify resistance hubs and context-sensitive sites.

Detailed implementations are provided in [seq_communication.ipynb](https://github.com/FahsaiNak/PLM-Epistasis/blob/main/jupyter_notebook/seq_communication.ipynb).

---

## Citation

If you use this framework, please cite:

Nakarin, F. et al. "An Interpretable Protein Language Model Framework Reveals Residue-Level Determinants of Viral Escape from Broadly Neutralizing Antibodies." (2025, in prep.)

---
