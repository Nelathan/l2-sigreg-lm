#!/usr/bin/env bash
set -euo pipefail

uv run ruff check src scripts
uv run ruff format --check src scripts
