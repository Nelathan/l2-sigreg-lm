# Notes

## Purpose

This file is the running lab notebook for the experiment.

Use it to capture:

- implementation progress
- spec clarifications
- training observations
- failures and dead ends
- hyperparameter decisions
- insights worth preserving

## Current Status

- 2026-03-25: Renamed the project and GitHub repo to `l2-sigreg-lm` so the title reflects the actual experiment: pure L2 embedding prediction plus SIGReg shaping, not token packing as the identity of the repo.
- 2026-03-25: Added linting as a pre-push gate via `scripts/lint.sh`, and the repo now passes `ruff check` plus `ruff format --check`.
- 2026-03-25: The strongest current L2 debug recipe is `embedding_init_std=0.01`, `prediction_head_init_std=0.0`, `l2_embedding_lr_scale=0.1`, `lambda_sigreg=0.05`, and SIGReg sampled over `2048` random vocab embeddings rather than only active document rows.
- 2026-03-25: Learned-output-scale and the mixed SIGReg pool over active-plus-random embeddings were both informative but not yet the right balance in the tested forms; the larger `0.8` init made SIGReg happier but hurt rank geometry, while the smaller `0.01` init preserved retrieval better.
- 2026-03-25: `MRR` remains the primary metric and `average_rank` is the best progress curve; `median_rank` is no longer needed for the active analysis loop.
- 2026-03-25: The next ablation is SIGReg sparsity, for example averaging over multiple `1024`-sample batches instead of a single harsh sample.
- 2026-03-25: The next optimization question is whether `l2_embedding_lr_scale` should stay at `0.1` or move back toward `1.0` once SIGReg is reduced.
- 2026-03-25: Performance tuning and broader benchmark work should wait until the 4070 Super box is available.
- 2026-03-25: Once SIGReg is stable, re-sweep `embedding_init_std` to find the new best radius for the embedding table.
- 2026-03-25: After the predictor stabilizes, test whether the head projector is still needed or whether it was mainly compensating for SIGReg applied to predictions.
- 2026-03-25: Revisit a mixed SIGReg token pool that includes some active tokens plus random vocab rows, to test whether active tokens also benefit from the isotropy pressure as a kind of rubber-band effect.
- 2026-03-25: Chosen working repo name is `l2-sigreg-lm`.
- 2026-03-25: Renamed the GitHub repository to `l2-sigreg-lm` so the project name matches the actual experiment.
- 2026-03-25: Added a lightweight lint step (`scripts/lint.sh`) and made it part of the pre-push lifecycle.
- 2026-03-24: Repository initialized from `SPEC.md`.
- 2026-03-24: Added project scaffold, agent instructions, and note-taking structure.
- 2026-03-24: `nanochat` is a reference for keeping the code path compact, but this repo remains smaller and optimized for M1 Pro iteration.
- 2026-03-24: Added the `lejepa` Git submodule at `submodules/lejepa`.
- 2026-03-24: Verified the package exposes `lejepa.multivariate.SlicingUnivariateTest` and `lejepa.univariate.EppsPulley`.
- 2026-03-24: Noted an API mismatch in `lejepa`: the README uses `num_points=17`, while the current `EppsPulley` implementation defines `n_points`.
- 2026-03-24: Implemented the first runnable training slice: explicit config presets, OpenWebText materialization for small debug runs, padded autoregressive batching with causal-plus-padding masking, a shared transformer with CE/L2 heads, retrieval metrics, basic embedding-health metrics, JSONL logging, and checkpointing.
- 2026-03-24: Deliberately started with padding instead of packed multi-document sequences to avoid cross-document attention leakage until segmented masking/position resets are designed properly.
- 2026-03-24: `uv sync` succeeds locally and generated `uv.lock`; the environment resolves with `torch==2.11.0`.
- 2026-03-24: Completed both `ce_smoke` and `l2_smoke` runs end to end and generated a first comparison report under `results/smoke_compare/`.
- 2026-03-24: Moved embedding-health computations to CPU-backed paths where needed because `svdvals` is not fully implemented for the required MPS path.
- 2026-03-24: Reworked nearest-neighbor collision monitoring to avoid constructing a full `V x V` similarity matrix in memory.
- 2026-03-24: Added cache-backed dataset materialization under `data/cache/`, keyed by the tokenization and budget config, so repeated runs can reuse the same local sequences deterministically.
- 2026-03-24: Replaced the cosine LR schedule with warmup/stable/decay. Default warmup is 5 percent of the run, and short smoke presets now use zero decay so the learning-rate curve does not confound brief diagnostics.
- 2026-03-24: Added train-time gradient diagnostics: global norm, max abs grad, embedding/head/trunk grad norms, active embedding-gradient row counts and density, mean active-row norm, and effective rank of the active embedding-gradient submatrix.
- 2026-03-24: Initial L2 smoke read suggests the embedding table dominates gradient magnitude relative to the prediction head and trunk, which is a useful clue for interpreting slow retrieval progress.
- 2026-03-24: Switched the project direction from English-only OpenWebText to multilingual `HuggingFaceFW/finewiki`, mixed across `en`, `ar`, `zh`, `fr`, `de`, `ja`, `ko`, and `es`, and standardized on the `lfm25` tokenizer.
- 2026-03-24: Fixed tokenizer-driven vocab sizing so model vocab now follows the active tokenizer, which was required for honest `lfm25` runs.
- 2026-03-24: Added byte-aware validation metrics, Harmax auxiliary evaluation for L2, and compare-script labels/plots that work for objective and ablation comparisons.
- 2026-03-24: Split L2 validation reporting into `val_pred_loss` and `val_sigreg_loss`; generic total validation loss was misleading because the SIGReg term dominated its scale.
- 2026-03-24: Retrieval evaluation now tracks `average_rank` in addition to `median_rank`, plus merged CE/Harmax rows in comparison reports.
- 2026-03-24: CE on the multilingual FineWiki+LFM setup learns quickly on smoke scale; L2 also learns, but with much weaker early signal and much stronger sensitivity to initialization.
- 2026-03-24: L2 init ablations:
  - Zeroing only the prediction head helped early retrieval relative to the original symmetric init.
  - A nanochat-inspired large embedding init with tiny head init was clearly worse for this geometric objective.
  - Best current L2 init is `embedding_init_std=0.01` with `prediction_head_init_std=0.0`.
- 2026-03-24: Promoted the winning L2 init into the default `l2_smoke` and `l2_debug` presets and removed the losing init ablations from the main preset surface.

### 2026-03-26 — GPU ablation sweep (4070 Super, 12 GB)

- What changed:
  - Added wandb integration with deferred init (no ghost runs), aggregated logging every 10 steps.
  - Ported data pipeline to nanochat-style: one-time tokenization to flat `.npy` files (`scripts/tokenize_data.py`), mmap'd `TokenStream` + `StreamingBatchIterator` with pre-allocated pinned CPU buffers.
  - Added EOT between documents, BOS at document start, pre-shuffled windows on disk.
  - Added bf16 mixed precision via `torch.autocast` — enables FlashAttention2 kernel via SDPA `is_causal=True`. No GradScaler needed (bf16 has fp32 exponent range). Master weights stay fp32.
  - Bypassed 4D attention mask construction when all tokens valid — SDPA flash kernel requires `is_causal=True` with no explicit mask.
  - Fixed critical bug: SIGReg was receiving transformer output predictions instead of token embeddings. Now `build_sigreg_inputs` looks up `model.token_embeddings(active_token_ids)`.
  - Added `pred_loss_scale` config to weight L2 prediction loss relative to SIGReg.
  - Added `use_prediction_head` toggle — backbone output can be used directly as L2 prediction.
- What was learned:
  - CE baseline at 5k steps: top1=34.7%, ppl=132, bpb=1.587.
  - L2+SIGReg with original recipe (emb=0.01, head=0.0, λ=0.05) completely collapsed on GPU scale — top1=0%, median_rank=29k. SIGReg dominated the loss.
  - **Embedding init scale is the primary knob.** Going from 0.01 → 1.0 reduced initial SIGReg from ~2400 to ~1, letting prediction drive learning.
  - **pred_loss_scale=10 is essential.** Without it, pred_loss is ~0.0001 and gets drowned by SIGReg.
  - **SIGReg must only regularize embeddings, never prediction vectors.** Feeding transformer outputs created extreme singular values.
  - **No prediction head beats having one.** Removing it: +1.4% top1, median_rank 191 vs 258.
  - **Embedding LR should be 1.0, not 0.1.** At 0.1x the embeddings couldn't respond to both SIGReg and prediction pressure.
  - Best recipe (`gpu_l2_5k_nohead`): top1=24.0%, median_rank=191, both losses alive and converging, still improving at 5k steps.
  - Training speed: 0.053s/step with bf16+flash (was 0.084s/step fp32, 0.2s/step with gradient SVD).
- What is still unclear:
  - Can L2 close the gap with CE on a longer run (20-50k steps)?
  - Optimal embedding_init_std — 1.0 was best tested but the sweep wasn't exhaustive.
  - Whether SIGReg warmup would help at this scale.
- Next step:
  - Long run (20-50k steps) with the nohead recipe.
  - Re-run CE baseline with new data pipeline for fair comparison.

## Open Questions

- Add resume support and explicit artifact naming so repeated longer runs do not overwrite each other's JSONL/checkpoints by default.
- Can L2+SIGReg close the gap with CE on a 20-50k step run?
- Would document-boundary masking (TRL-style packing with block-diagonal attention via FlexAttention) improve results enough to justify the complexity?
- Optimal embedding_init_std sweep — 1.0 was best so far but not exhaustively tested.
- Whether SIGReg warmup would help at GPU scale.

## Nanochat Design Takeaways

### What to Borrow

- Keep the codebase compact and directly runnable. `nanochat` stays understandable because the main training path is easy to trace from script to model to dataloader.
- Use a small number of files with clear responsibilities instead of building a framework.
- Prefer explicit runtime/device handling over hidden magic. A small `common` module for device detection, seeding, and logging is worth copying.
- Keep the trainer script in charge of orchestration, with the model and dataloader modules staying relatively dumb.
- Use simple shell entrypoints for canonical runs, including one clearly small Apple Silicon path.
- Make evaluation artifacts and checkpoints first-class local outputs.

### What Not to Borrow

- Do not import the multi-stage pipeline shape. This project is pretraining-style retrieval research only, with no tokenizer training, SFT, RL, inference engine, or UI.
- Do not adopt the "one complexity dial" philosophy. Our architecture is fixed by the experiment spec; the main variable is objective function and a small number of research hyperparameters.
- Do not bring in distributed-training assumptions, wandb, or heavyweight optimization machinery unless the experiment proves it needs them.
- Do not mirror nanochat's breadth. We want fewer moving parts than nanochat, not a reduced copy of it.

### Concrete Repo Direction

- Keep one shared transformer implementation with two heads in `src/model.py`.
- Keep one training entrypoint that switches between `l2_sigreg` and `ce_baseline`.
- Keep data packing and validation ordering deterministic so comparisons are strict.
- Keep logging local and structured, likely JSONL plus a small comparison script.
- Add one tiny M1-friendly debug run path before any full experiment path.

## Design Ripple Notes

- Packed documents are only valid if we preserve document boundaries in attention behavior.
- If multiple documents share one training sequence, we must prevent cross-document causal attention.
- That likely means carrying document segment metadata from the packer into the model and using position-id resets plus a document-aware causal mask or equivalent flash-attention-compatible trick.
- This is not a small dataloader-only optimization; it affects model inputs, evaluation consistency, and any future attempt to swap attention implementations.
- Before implementing packing, decide explicitly between:
  - simpler baseline: one contiguous token stream with `<endoftext>` separators and no boundary masking
  - stricter packed-doc path: packed sequences with boundary-aware positions and blocked cross-document attention

## Progress Log Template

### YYYY-MM-DD

- What changed:
- What was learned:
- What is still unclear:
- Next step:
