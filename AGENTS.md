# AGENTS

## Purpose

`token-packing-lab` is a deliberately compact language-model training experiment on Apple Silicon. The goal is to compare:

- `l2_sigreg`: transformer trained to predict the next-token embedding with L2 loss plus SIGReg regularization
- `ce_baseline`: same transformer trunk trained with tied-embedding cross-entropy

`SPEC.md` is the source of truth. If code and spec diverge, align code to the spec unless an explicit decision has been recorded in `docs/NOTES.md`.

## Working Principles

- Keep the implementation small, readable, and M1 Pro friendly.
- Prefer a nanochat-style code shape: few files, direct training loops, minimal abstractions, easy inspection.
- Use `uv` for environment and dependency management.
- Do not add external services or hosted logging.
- Do not reimplement SIGReg. Use the published `lejepa` submodule.
- Keep both experiment branches as identical as possible except for the prediction head and loss.

## Initial Build Priorities

1. Create a minimal, working training stack for tokenized packed sequences.
2. Implement one shared transformer trunk with switchable heads.
3. Add evaluation based on retrieval ranking, not generation.
4. Add embedding-health monitoring for the L2 setup.
5. Keep logs and checkpoints simple and local.

## Repository Conventions

- Put experiment code in `src/`.
- Put runnable entrypoints and comparison utilities in `scripts/`.
- Put durable notes, findings, and deviations from the spec in `docs/NOTES.md`.
- Save generated artifacts to `results/`.
- Keep scripts non-interactive and reproducible.

## Development Lifecycle

1. Keep the first pass small and debuggable.
2. Validate each meaningful change with a short preset before scaling it up.
3. Prefer one variable at a time for ablations unless the user explicitly wants a coupled sweep.
4. Run `uv run bash scripts/lint.sh` before committing or pushing.
5. Treat `docs/NOTES.md` as the durable lab notebook and `README.md` as the clean developer-facing summary.
6. Commit once the docs, code, and experiment state agree on the next step.

## Operating Notes

- Target runtime is Apple M1 Pro with MPS.
- Default assumptions should optimize for low memory pressure and resumable runs.
- During early development, short debug runs are preferred over full-scale jobs.
- Add assertions for the L2 target-embedding gradient behavior during development, then remove or gate them for long runs.

## Suggested Commands

- `uv sync`
- `uv run python -m src.train --config l2_debug`
- `uv run python -m src.train --config ce_debug`
- `uv run python scripts/compare.py`
