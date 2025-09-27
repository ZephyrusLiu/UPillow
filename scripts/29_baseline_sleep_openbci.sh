#!/usr/bin/env bash
# Simple baseline pipeline for projecting 16-channel OpenBCI recordings onto
# Sleep-EDF style Fpz-Cz / Pz-Oz derivations, running the pretrained CNN, and
# optionally evaluating predictions against reference labels.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <openbci_input.(txt|csv)> [labels.npy|labels.csv]" >&2
  exit 1
fi

INPUT_PATH="$1"
LABELS_PATH="${2:-}"

OUT_DIR="outputs"
OUT_FILE="${OUT_DIR}/sleep_preds_baseline.npy"

mkdir -p "${OUT_DIR}"

python -m src.sleep.infer_sleep_openbci \
  --input "${INPUT_PATH}" \
  --baseline-aliases configs/openbci_baseline_aliases.json \
  --disable-car \
  --out "${OUT_FILE}"

if [[ -n "${LABELS_PATH}" ]]; then
  python -m src.sleep.eval_sleep_openbci \
    --preds "${OUT_FILE}" \
    --labels "${LABELS_PATH}"
else
  echo "No labels provided; skipping accuracy evaluation." >&2
fi