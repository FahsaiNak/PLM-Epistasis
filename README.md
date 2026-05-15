# Decoding Viral Escape using Attention Maps from Protein Language Models

We fine-tune a protein language model (**ESM-2**) to predict HIV-1 Env (gp160) neutralization sensitivity to the broadly neutralizing antibody **VRC01**, and use **co-attention** and **attribution (Integrated Gradients)** analyses to interpret the molecular basis of viral escape.

---

## Overview

| Stage | Description |
|:------|:-------------|
| **Model training** | Fine-tune ESM on labeled gp160 sequences (sensitive vs. resistant to VRC01) |
| **Model evaluation** | Evaluate model performance and save per-sequence predictions |
| **Attention extraction** | Compute [attention rollout](https://doi.org/10.48550/arXiv.2005.00928) to identify residue–residue relationships |
| **Attribution analysis** | Use [Integrated Gradients (Captum)](https://captum.ai/docs/extension/integrated_gradients) to estimate residue-level contributions to class predictions |
| **Residue communication** | Construct importance-weighted directional communication networks from attention scores |
| **Epistatic coupling** | Learn context-dependent coupling coefficients via ElasticNet regression on communication-weighted features |
| **Fitness inference** | Compute the antigen-antibody fitness score E(σ) integrating field and coupling terms |

---

## Workflow Summary

This project decodes the molecular basis of **HIV-1 escape from the broadly neutralizing antibody VRC01** using **ESM-based protein language models** and **co-attention attribution** analysis.

The full workflow consists of seven major stages:

---

### **1. Data Preparation**
**Goal:** Build a curated dataset linking HIV-1 gp160 sequences to VRC01 neutralization outcomes.

**Steps:**
- Collect full-length HIV-1 Env gp160 amino acid sequences from the [LANL HIV Sequence Database](https://www.hiv.lanl.gov/).
- Gather experimental IC80 values (neutralization potency) from the [CATNAP database](https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/).
- Label each sequence as:
  - `1` → **VRC01-sensitive** (IC80 < 1.0 µg/mL; 306 strains)
  - `0` → **VRC01-resistant** (remaining 583 strains)
- Save formatted data in `data/input_VRC01_IC80.csv`.

**Output:**  
`input_VRC01_IC80.csv` — input dataset for model training and evaluation.

---

### **2. Model Fine-Tuning**
**Goal:** Train a transformer-based classifier to distinguish VRC01-sensitive vs. resistant sequences.

**Method:**
- **Backbone:** `facebook/esm2_t33_650M_UR50D`
- **Classifier head:** 2-layer MLP (`hidden_dim = 128`)
- **Pooling:** `[CLS]` token
- **Unfrozen layers:** Top 9 ESM-2 transformer layers (lower layers frozen to preserve general evolutionary knowledge)
- **Regularization:** Random token masking (15%), dropout (0.3), early stopping (patience = 10 epochs)
- **Loss:** CrossEntropy
- **Optimizer:** AdamW (`lr = 1.68e-5`)
- **Scheduler:** Linear warmup

**Procedure:**
- Optimize hyperparameters using [Optuna](https://optuna.org/) with 5-fold cross-validation.
- Train an ensemble of **5 independent model replicates**, each initialized with a different random seed, for robustness. Variance across replicates provides confidence intervals for predictions and interpretability scores.
- Save trained weights upon best validation loss (early stopping with patience = 10 epochs, max 50 epochs).

**Run:**
```bash
python src/run_full_train.py \
    --task_type classification --criterion CrossEntropy \
    --dropout 0.3 --head_hidden_dim 128 \
    --pooling cls --unfreeze_layers 9 \
    --batch_size 6 --lr 1.68e-5 \
    --epochs 50 \
    --input_csv data/input_VRC01_IC80.csv
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
- Extract raw attention arrays `A(ℓ,h)` from all ESM-2 transformer layers `ℓ` and heads `h`.
- Apply [attention rollout](https://doi.org/10.48550/arXiv.2005.00928) and average across heads.
- Compare patterns between sensitive and resistant groups to detect co-evolving residues.

**Run:**
```bash
python src/run_attentions.py --task_type classification --input_csv data/input_VRC01_IC80.csv
```

**Output:**  
`results/attention_maps/classification/rep_{i}/` — per-sequence attention matrices (`.npy`).

---

### **5. Attribution Analysis (Integrated Gradients)**
**Goal:** Quantify residue-level contributions to model predictions.

**Method:**
- Apply **[Integrated Gradients (Captum)](https://captum.ai/docs/extension/integrated_gradients)** to compute signed per-residue attribution scores `ϕ̄(k,i)`.
- Run for both target classes:
  - `target_class = 1` → VRC01-sensitive
  - `target_class = 0` → VRC01-resistant
- Aggregate attribution results across replicates (ensemble mean) for stability.
- Identify residues whose attributions shift significantly by amino acid identity using two-sample t-tests.

**Run:**
```bash
python src/run_attributions.py --task_type classification --target_class 0 --input_csv data/input_VRC01_IC80.csv
python src/run_attributions.py --task_type classification --target_class 1 --input_csv data/input_VRC01_IC80.csv
```

**Output:**  
`results/attribution_maps/classification/rep_{i}/` — residue-level importance scores.

---

### **6. Residue Communication Map**
**Goal:** Map how information propagates between residues within the model.

**Method:**
Construct a directed communication network by weighting raw residue–residue attention with each residue's global importance to the classification decision. For each transformer layer `ℓ` and attention head `h`:

1. Separate the residue-to-residue interaction submatrix `A_RR ∈ ℝ^(N×N)` and the `<cls>`-to-residue attention vector `A_cls ∈ ℝ^N` (min-max normalized).
2. Weight each interaction by the source residue's normalized importance:

   `M(ℓ,h)_(i,j) = A(ℓ,h)_RR(i,j) × A(ℓ,h)_cls(i)`

3. Sum across all layers and heads to obtain the final communication map `M`, where `M_(i,j)` captures importance-weighted information flow **from residue j to residue i**.

Detailed implementations are provided in [seq_communication.ipynb](https://github.com/FahsaiNak/PLM-Epistasis/blob/main/jupyter_notebook/seq_communication.ipynb).

---

### **7. Learning Epistatic Coupling**
**Goal:** Identify context-dependent interactions between residues that jointly shape VRC01 neutralization.

**Method:**
For each target state `(i, a)` (position `i` carrying amino acid `a`):

1. Collect training sequences `S_{i,a} = {k : σ^(k)_i = a}`.
2. Define the response as the deviation from the state median:

   `Y_k = ϕ̄_(k,i) − median_{k' ∈ S_{i,a}}(ϕ̄_(k',i))`

3. Fit a ridge-dominated **ElasticNet** regression (with 5-fold CV alpha selection, 1000 bootstrap iterations for stability selection, and 500 permutation tests for empirical p-values) regressing `Y_k` onto communication-weighted one-hot features:

   `X_(k, j, b) = M_(k, i←j) × 𝟙[σ^(k)_j = b]`

4. The resulting coefficients `c^(j,b)_(i,a)` form the **epistatic coupling map**, encoding how amino acid `b` at source position `j` — weighted by its information flow to target `i` — modulates the neutralization attribution at `(i, a)`.

**Run:**
```bash
python src/fit_coupling.py
```

**Output:**  
`results/coupling/epistatic_map.pkl` — sparse dictionary of coupling weights, bootstrap confidence scores, permutation p-values, and sequence support counts.

---

### **8. Model Inference for Antigen-Antibody Fitness**
**Goal:** Compute a sequence-specific antigen-antibody fitness score that integrates field and epistatic coupling contributions.

**Method:**
The fitness score `E(σ)` for an Env sequence `σ = (σ_1, …, σ_L)` is computed as:

`E(σ) = Σ_i [ H(i, σ_i) + C_i(σ) ]`

- **Field term** — intrinsic, context-independent propensity of amino acid `a` at position `i`:

  `H(i, a) = median_{k ∈ S_{i,a}}( ϕ̄_(k,i) )`

- **Coupling term** — epistatic influence of all source positions, gated by the sequence-specific communication map:

  `C_i(σ) = Σ_j  M_(σ, i←j) · c^(j, σ_j)_(i, σ_i)`

A more **positive** `E` indicates VRC01 sensitivity; **negative** values signal resistance-conferring configurations.

Detailed implementations are provided in [AMP_analysis.ipynb](https://github.com/FahsaiNak/PLM-Epistasis/blob/main/jupyter_notebook/AMP_analysis.ipynb) and [fit_coupling.ipynb](https://github.com/FahsaiNak/PLM-Epistasis/blob/main/jupyter_notebook/fit_coupling.ipynb).

---

## Citation

If you use this framework, please cite:

Nakarin, F. et al. "Decoding HIV-1 Antibody Escape with Interpretable Protein Language Models." (2025, in prep.)

---
