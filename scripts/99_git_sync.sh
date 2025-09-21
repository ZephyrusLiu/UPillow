#!/usr/bin/env bash

# Lightweight wrapper to invoke the GitHub sync helper.
#
# Usage examples:
#   bash scripts/99_git_sync.sh push --set-upstream
#   bash scripts/99_git_sync.sh pull --remote origin --branch main

set -euo pipefail

python -m src.tools.github_sync "$@"