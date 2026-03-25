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

## Open Questions

- Confirm the best `PyTorch` release usable with Python 3.14 on Apple Silicon before first environment lock.
- Decide whether to cache tokenized OpenWebText to memory-mapped binary shards or a simpler local format for the first pass.
- Decide how strict the L2 embedding-gradient development assertion should be once the final batching strategy is locked.
- Add a true global-median rank computation during validation instead of the current per-batch approximation.
- Add resume support and explicit artifact naming so repeated longer runs do not overwrite each other's JSONL/checkpoints by default.
- Decide whether SIGReg should have an explicit warmup schedule now that the base L2 init is more stable.
- Decide whether to reduce L2 embedding LR after measuring gradient flow on a longer run with the new default init.

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
