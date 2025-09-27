"""Evaluate OpenBCI sleep staging predictions against reference labels."""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def _load_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.extend(part for part in line.replace(",", " ").split(" ") if part)
    return np.asarray([int(x) for x in data], dtype=np.int64)


def _parse_labels(names: str) -> Sequence[str]:
    return [name.strip() for name in names.split(",") if name.strip()]


def _print_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> None:
    labels = list(range(len(class_names)))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print("Cohen Kappa:", cohen_kappa_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred, labels=labels))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )


def _validate_lengths(preds: Iterable[int], labels: Iterable[int]) -> None:
    if len(preds) != len(labels):
        raise ValueError(
            f"Length mismatch: got {len(preds)} predictions but {len(labels)} labels"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Path to numpy/text file with predicted classes")
    ap.add_argument("--labels", required=True, help="Path to numpy/text file with reference labels")
    ap.add_argument(
        "--class-names",
        default="W,N1,N2,N3,REM",
        help="Comma separated class names to print in the report",
    )
    args = ap.parse_args()

    preds = _load_array(args.preds).astype(np.int64)
    labels = _load_array(args.labels).astype(np.int64)
    _validate_lengths(preds, labels)

    class_names = _parse_labels(args.class_names)
    n_classes = int(max(preds.max(), labels.max()) + 1)
    if not class_names or len(class_names) < n_classes:
        class_names = [f"Class {i}" for i in range(n_classes)]

    _print_metrics(labels, preds, class_names)


if __name__ == "__main__":
    main()