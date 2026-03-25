# L2 Embedding Packing Experiment

## Purpose

Test whether gradient-based training can pack a large vocabulary into a low-dimensional embedding space using L2 prediction loss + SIGReg regularization, and whether the resulting model retrieves correct tokens competitively with a cross-entropy baseline.

This is NOT a generation experiment. No sampling, no text output. Pure retrieval evaluation: given a context, does the predicted embedding land near the right token?

## Research Question

Can a token-level autoregressive language model trained with L2 loss in d=128 embedding space (no softmax, no cross-entropy, no V×d backward fan-out) learn useful vocabulary packing for V=50,257 tokens, as measured by retrieval metrics against an identical CE-trained baseline?

---

## Project Setup

### Toolchain

- **Python**: 3.14
- **Package manager**: `uv` — all dependencies managed via `uv`
- **PyTorch**: 2.11 (MPS backend for M1 Pro development)
- **Hardware target**: Apple M1 Pro, 32GB RAM. All models must fit comfortably in memory. No CUDA required for this experiment.

### Repository Structure

```
l2-packing-experiment/
├── pyproject.toml
├── uv.lock
├── submodules/
│   └── lejepa/              # SIGReg implementation
├── src/
│   ├── model.py             # Transformer + both heads
│   ├── data.py              # Dataset loading and tokenization
│   ├── train.py             # Training loop for both configurations
│   ├── eval.py              # Retrieval metrics + Boltzmann pseudo-perplexity
│   ├── monitor.py           # Embedding health monitoring
│   └── config.py            # All hyperparameters
├── scripts/
│   ├── run_l2.sh
│   ├── run_ce.sh
│   └── compare.py           # Side-by-side metric comparison
└── results/
    └── .gitkeep
```

### Git Submodules

```
[submodule "submodules/lejepa"]
    path = submodules/lejepa
    url = https://github.com/galilai-group/lejepa.git
```

Import SIGReg from the lejepa package. The core loss is in `lejepa.multivariate.SlicingUnivariateTest` with `lejepa.univariate.EppsPulley` as the test statistic. If the import path differs, search the repo for `SlicingUnivariateTest` and adapt. Do NOT reimplement SIGReg — use the published code.

---

## Model Architecture

A single transformer architecture shared by both configurations. The ONLY difference between L2 and CE is the prediction head.

### Shared Trunk

| Parameter         | Value     | Notes                                    |
|-------------------|-----------|------------------------------------------|
| Layers            | 6         | Standard pre-norm transformer blocks     |
| Hidden dim (d)    | 128       | This is the variable under test          |
| Attention heads   | 4         | Head dim = 32                            |
| FFN dim           | 512       | 4× hidden dim                            |
| Sequence length   | 512       | Tokens per training example              |
| Vocabulary (V)    | 50,257    | GPT-2 tokenizer, `tiktoken` encoding     |
| Positional encoding | RoPE    | Standard rotary embeddings               |
| Normalization     | RMSNorm   | Pre-norm placement                       |
| Dropout           | 0.0       | Not needed at this scale                 |
| Total params      | ~8M       | Verify this lands in 6-10M range         |

### Configuration A: L2 + SIGReg

- **Input embedding matrix** `E ∈ ℝ^{V×d}`: maps token IDs to d-dimensional embeddings. This is the vocabulary embedding table.
- **Prediction head**: a single linear projection `W_pred ∈ ℝ^{d×d}` applied to the transformer output at each position. Output is a d-dimensional predicted embedding.
- **Training loss**:
  ```
  L_total = L_pred + λ * SIGReg(Z)

  L_pred = mean over positions of ‖ W_pred(h_t) - E[target_t] ‖²
  ```
  Where `h_t` is the transformer output at position t, `E[target_t]` is the embedding of the ground-truth next token looked up from the SAME embedding matrix `E`, and `Z` is the batch of predicted embeddings.
- **SIGReg**: applied to the matrix of all predicted embeddings in the batch (batch_size × seq_len, d). Use `num_slices=1024` and `EppsPulley(num_points=17)` as defaults from the LeJEPA repo. If these defaults cause issues at d=128, try `num_slices=512`.
- **λ**: start at 1.0, sweep {0.1, 0.5, 1.0, 2.0, 5.0} in a short hyperparameter search (2000 steps each).
- **CRITICAL**: The gradient from `L_pred` flows to `W_pred`, through the transformer trunk, AND to the target embedding `E[target_t]`. The embedding table `E` receives gradient from being a prediction target. It does NOT receive gradient through any V-dimensional fan-out. This is the entire point. Verify this by checking that `E.grad` has nonzero entries only for tokens that appeared as targets in the batch. Add an assertion for this during development, remove it for training runs.

### Configuration B: CE Baseline

- **Same trunk**, same embedding matrix `E`, same positional encoding.
- **Prediction head**: standard unembedding `logits = h_t @ E.T`, producing V-dimensional logits.
- **Training loss**: standard cross-entropy over logits.
- **No SIGReg**.
- **Shared embedding**: tie input embedding and output unembedding (weight tying), as is standard practice. This means the L2 model and CE model have the same parameterization of `E`, just trained differently.

### Weight Initialization

Both configurations: standard transformer init. Embedding matrix `E` initialized from `N(0, 0.02)`. This is important — both models start from the same `E` distribution, so differences in final embedding structure are attributable to training dynamics, not initialization.

Use the SAME random seed for both configurations. Identical initialization.

---

## Dataset

### Source

OpenWebText via HuggingFace: `Skylion007/openwebtext`

### Preprocessing

- Tokenize with `tiktoken`, encoding `gpt2` (produces V=50,257 vocabulary).
- Pack into sequences of 512 tokens. Concatenate documents with an `<endoftext>` separator. No padding.
- **Train split**: first 1B tokens (after tokenization). This is ~2M sequences of length 512.
- **Validation split**: next 10M tokens (~20k sequences).
- **IMPORTANT**: Both configurations train on the EXACT same data in the EXACT same order. Use a fixed seed for the dataloader.

### Dataloading

Standard PyTorch DataLoader with prefetching. Batch size 64 (= 64 × 512 = 32,768 tokens per step). Adjust if memory is tight on M1 Pro — minimum batch size 32.

---

## Training

### Optimizer

- AdamW, weight decay 0.1
- Learning rate: 3e-4 with cosine schedule, linear warmup for first 2000 steps
- β1=0.9, β2=0.95
- Gradient clipping: max norm 1.0

### Duration

- 50,000 steps (~1.6B tokens seen, ~1.6 epochs over the 1B token train set)
- Validate every 1000 steps
- Checkpoint every 5000 steps

### Logging

Use a simple JSON-lines log file. One line per validation step with all metrics. No W&B or external dependencies.

---

## Evaluation

Evaluate on the validation split at each validation step.

### Primary Metrics (computed for BOTH configurations)

For each position in each validation sequence, compute the score of every vocabulary token and rank them.

**For L2 model**: score_i = -‖ z_pred - E[i] ‖² for all i ∈ V. Lower distance = higher rank.

**For CE model**: score_i = logit_i (the pre-softmax output). Higher logit = higher rank.

Then compute:

1. **Top-1 Accuracy**: fraction of positions where the highest-ranked token is the ground truth.
2. **Top-5 Accuracy**: fraction where ground truth is in the top 5.
3. **Top-10 Accuracy**: fraction where ground truth is in the top 10.
4. **MRR (Mean Reciprocal Rank)**: mean of 1/rank(ground_truth) across all positions.
5. **Median Rank**: median rank of the ground-truth token.

### Secondary Metric

6. **Boltzmann Pseudo-Perplexity** (L2 model only): convert L2 distances to a probability distribution via:
   ```
   p_i = exp(-‖z_pred - E[i]‖² / τ) / Σ_j exp(-‖z_pred - E[j]‖² / τ)
   ```
   with τ chosen to minimize perplexity on a held-out calibration set (first 1000 validation sequences). Then compute standard perplexity on the remaining validation data. This is purely a measurement device — the training never sees this computation. Report the calibrated τ alongside the perplexity.

   For the CE model, compute standard perplexity from the softmax distribution.

### Embedding Health Monitoring (L2 model only)

Compute these every validation step and log them:

7. **Embedding matrix singular values**: compute SVD of `E`, log the full spectrum (all 128 singular values). Plot as a line chart per checkpoint. Watch for rank collapse (sudden drop-off) or uniform spread (healthy).
8. **Average pairwise cosine similarity**: sample 10,000 random pairs from `E`, compute mean cosine similarity. Should stay near 0 if SIGReg is working. If it drifts toward 1, embeddings are collapsing.
9. **Effective dimensionality**: participation ratio of the singular values: `(Σ σ_i)² / Σ σ_i²`. Ranges from 1 (rank-1 collapse) to d (fully isotropic). Track over training.
10. **Nearest-neighbor collision rate**: for each token embedding, find its nearest neighbor. Count how many tokens share the same nearest neighbor. High collision rate = degenerate packing.

---

## Comparison Script

`scripts/compare.py` loads the final checkpoint of both configurations and produces a single markdown table:

```
| Metric                    | L2 + SIGReg | CE Baseline | Δ        |
|---------------------------|-------------|-------------|----------|
| Top-1 Accuracy            |             |             |          |
| Top-5 Accuracy            |             |             |          |
| Top-10 Accuracy           |             |             |          |
| MRR                       |             |             |          |
| Median Rank               |             |             |          |
| Pseudo-Perplexity (L2) / Perplexity (CE) |  |          |          |
| Effective Dimensionality  |             | N/A         |          |
| Avg Cosine Similarity     |             | N/A         |          |
```

Also generate:
- A plot of top-1 accuracy over training steps for both models (same axes).
- A plot of the embedding singular value spectrum at final checkpoint for both models.
- A histogram of ground-truth token ranks for both models at final checkpoint (log scale x-axis).

Save all plots as PNG to `results/`.

---

## What This Experiment Proves or Disproves

**If L2 + SIGReg top-1 accuracy is within 5% of CE baseline**: vocabulary packing works at d=128. The path to L2-trained LMs is open. The next step is scaling up and adding a sampling mechanism.

**If L2 + SIGReg top-5 accuracy is competitive but top-1 is significantly worse**: the model learns good neighborhoods but can't pinpoint exact tokens. This is expected and fine — it means temperature-controlled sampling (Gumbel-max) would produce diverse, valid text.

**If the embedding singular value spectrum collapses during training**: SIGReg λ is too low or the loss landscape is inherently degenerate at this d/V ratio. Sweep λ before concluding.

**If top-1 accuracy is catastrophically worse (>20% gap)**: vocabulary packing at d=128 is not learnable by gradient descent for natural language. Document the failure mode — which tokens collide, whether rare or frequent tokens fail, whether the spectrum shows the problem. This negative result is publishable.

---

## Implementation Notes

- Use `torch.cdist` for batched L2 distance computation during eval. For training, the L2 loss is just MSE between two d-dimensional vectors — do not compute distances to all V tokens during training.
- The full V×d distance matrix during eval will be 50,257 × 128 × 4 bytes ≈ 25MB per batch element. Compute it in chunks if memory is tight.
- `tiktoken` is faster than HuggingFace tokenizers. Use it.
- Do NOT use `torch.compile` on MPS — it's unreliable. Use eager mode.
- MPS does not support all operations. If something fails on MPS, fall back to CPU for that operation (likely SVD in monitoring). Keep training on MPS.
