"""Baseline utilities for projecting 16-channel OpenBCI layouts to Sleep-EDF pairs."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


CANONICAL_REGIONS: Mapping[str, Tuple[str, ...]] = {
    # Approximations for canonical Sleep-EDF derivations.
    # These tuples are ordered by preference (front of tuple = highest priority).
    "fpz": ("Fpz", "Fp1", "Fp2", "AFz", "AF3", "AF4"),
    "cz": ("Cz", "CPz", "C3", "C4"),
    "pz": ("Pz", "CPz", "P3", "P4"),
    "oz": ("Oz", "O2", "O1"),
}


def _normalize(name: str) -> str:
    """Normalize channel labels for case-insensitive comparison."""
    return name.strip().lower()


def _invert_aliases(
    aliases: Mapping[str, Sequence[str]] | None
) -> MutableMapping[str, List[str]]:
    """Merge user-provided aliases with the default canonical mapping."""
    merged: MutableMapping[str, List[str]] = {
        key: list(values) for key, values in CANONICAL_REGIONS.items()
    }
    if not aliases:
        return merged
    for canonical, extra in aliases.items():
        key = canonical.strip().lower()
        if key not in merged:
            merged[key] = []
        merged[key].extend(extra)
    return merged


def _resolve_candidates(
    channel_names: Sequence[str],
    candidates: Iterable[str],
) -> List[int]:
    """Return indices of all channels whose name matches candidates."""
    available = {_normalize(name): idx for idx, name in enumerate(channel_names)}
    resolved: List[int] = []
    for cand in candidates:
        key = _normalize(cand)
        if key in available and available[key] not in resolved:
            resolved.append(available[key])
    return resolved


def _average_channels(X: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    if not indices:
        raise ValueError("No channels available for requested region")
    if len(indices) == 1:
        return X[indices[0]].astype(np.float32)
    return X[np.array(indices)].mean(axis=0).astype(np.float32)


def project_to_sleepedf_pairs(
    X: np.ndarray,
    channel_names: Sequence[str],
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> Tuple[np.ndarray, Dict[str, Sequence[str]]]:
    """Project 16-channel OpenBCI signals onto Sleep-EDF style Fpz-Cz / Pz-Oz pairs.

    Parameters
    ----------
    X:
        Array of shape (C, T) containing cleaned OpenBCI signals.
    channel_names:
        Sequence of channel labels aligned with the first dimension of ``X``.
    aliases:
        Optional mapping that augments :data:`CANONICAL_REGIONS` with project-specific
        aliases (e.g., ``{"fpz": ["EXG0"]}``). The lookup is case-insensitive.

    Returns
    -------
    projected, meta
        ``projected`` is a ``(2, T)`` array with ``[0]`` approximating Fpz-Cz and
        ``[1]`` approximating Pz-Oz. ``meta`` reports which physical channels were
        used for each virtual derivation.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (channels, samples)")
    merged = _invert_aliases(aliases)

    fpz_idx = _resolve_candidates(channel_names, merged["fpz"])
    cz_idx = _resolve_candidates(channel_names, merged["cz"])
    pz_idx = _resolve_candidates(channel_names, merged["pz"])
    oz_idx = _resolve_candidates(channel_names, merged["oz"])

    if not fpz_idx:
        raise ValueError("Could not locate any channels for Fpz approximation")
    if not cz_idx:
        raise ValueError("Could not locate any channels for Cz reference")
    if not pz_idx:
        raise ValueError("Could not locate any channels for Pz approximation")
    if not oz_idx:
        raise ValueError("Could not locate any channels for Oz reference")

    fpz_signal = _average_channels(X, fpz_idx)
    cz_signal = _average_channels(X, cz_idx)
    pz_signal = _average_channels(X, pz_idx)
    oz_signal = _average_channels(X, oz_idx)

    projected = np.stack(
        [fpz_signal - cz_signal, pz_signal - oz_signal], axis=0
    ).astype(np.float32)
    meta = {
        "Fpz-Cz": {
            "positive": [channel_names[i] for i in fpz_idx],
            "reference": [channel_names[i] for i in cz_idx],
        },
        "Pz-Oz": {
            "positive": [channel_names[i] for i in pz_idx],
            "reference": [channel_names[i] for i in oz_idx],
        },
    }
    return projected, meta


__all__ = [
    "project_to_sleepedf_pairs",
    "CANONICAL_REGIONS",
]
