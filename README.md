# Token Packing Lab

Small-scale training repo for testing whether an autoregressive transformer can learn useful token retrieval with an L2 embedding objective plus SIGReg regularization, compared against a matched cross-entropy baseline.

The authoritative project definition lives in `SPEC.md`.

## Status

Core training, evaluation, and comparison code is in place. Current work is on ablations, longer runs, and tightening the L2 geometry story.

## Development Lifecycle

1. Sync the environment with `uv sync`.
2. Run a short debug preset first, usually `l2_debug` or `ce_debug`.
3. Compare the most recent runs with the local comparison script and review the plots.
4. Run `uv run bash scripts/lint.sh` before pushing changes.
5. Record durable findings, hyperparameter decisions, and spec deviations in `docs/NOTES.md`.
6. Keep changes small and local until the current hypothesis is resolved.

## Planned Layout

See `SPEC.md` for the detailed experiment setup and `docs/NOTES.md` for the running research log.
