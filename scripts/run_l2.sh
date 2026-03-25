#!/usr/bin/env bash
set -euo pipefail

uv run python -m src.train --config l2_debug "$@"

