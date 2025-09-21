import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "sigma": (12, 16),
    "beta": (16, 30)
}

def band_powers(x: np.ndarray, fs: int):
    """x: (T,) single-epoch z-scored"""
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    feats = []
    total = np.trapz(Pxx, f) + 1e-12
    for (lo, hi) in BANDS.values():
        mask = (f >= lo) & (f <= hi)
        bp = np.trapz(Pxx[mask], f[mask])
        feats.append(bp / total)
    return np.array(feats, dtype=np.float32)
