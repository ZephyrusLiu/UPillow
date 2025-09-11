import os, re, csv, json
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample
from typing import Tuple, Dict, Any, Optional

OPENBCI_SAT_VALS = set([
    -187500.02235174447, -93750.01117587223, -8388608, 8388607
])

def _read_openbci_txt(path: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Read OpenBCI GUI .txt (RAW) file with header lines starting by '%'
    Returns: data (N,T), fs, meta
    """
    meta = {}
    headers = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        # collect header lines beginning with %
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.startswith('%'):
                f.seek(pos); break
            headers.append(line.strip())
    # parse metadata
    for h in headers:
        if 'Number of channels' in h:
            try:
                meta['n_channels'] = int(re.search(r'=\s*(\d+)', h).group(1))
            except: pass
        if 'Sample Rate' in h:
            try:
                meta['fs'] = float(re.search(r'=\s*([0-9.]+)', h).group(1))
            except: pass
        if 'Board' in h:
            meta['board'] = h.split('=')[-1].strip()
    # read table
    # Expect columns starting "Sample Index, EXG Channel 0, ..., EXG Channel 15, ..."
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('%') or not line.strip():
                continue
            rows.append(line.strip().split(','))
    # header line (column names) is first data row?
    # Detect if first row is non-numeric → treat as header
    def _is_number(s):
        try:
            float(s); return True
        except:
            return False
    if rows and not _is_number(rows[0][0]):
        colnames = [c.strip() for c in rows[0]]
        data_rows = rows[1:]
    else:
        colnames = None
        data_rows = rows
    # collect EXG columns 0..15 if available
    exg_idxs = []
    if colnames:
        for i, name in enumerate(colnames):
            if name.strip().lower().startswith('exg channel'):
                exg_idxs.append(i)
    else:
        # assume columns 1..16 are EXG when no titles (after "Sample Index")
        exg_idxs = list(range(1, 17))
    mat = []
    for r in data_rows:
        if len(r) < (max(exg_idxs)+1):
            continue
        try:
            mat.append([float(r[i]) for i in exg_idxs])
        except:
            continue
    X = np.asarray(mat, dtype=np.float64).T  # (C,T)
    fs = float(meta.get('fs', 125.0))
    meta['channel_names'] = [f'EXG{i}' for i in range(X.shape[0])]
    return X, fs, meta

def _read_brainflow_csv(path: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Read BrainFlow CSV export (assumed columns with EXG data)."""
    # naive CSV read: find 16 columns that look like EXG signals (float, wide range)
    import pandas as pd
    df = pd.read_csv(path)
    # Heuristic: pick first 16 float columns excluding obvious timestamps
    float_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    # common timestamp cols
    ts_cols = [c for c in float_cols if 'timestamp' in str(c).lower() or 'time' in str(c).lower()]
    cand = [c for c in float_cols if c not in ts_cols][:16]
    X = df[cand].to_numpy(dtype=np.float64).T
    fs = 125.0  # fallback; adjust if metadata available
    meta = {'n_channels': X.shape[0], 'fs': fs, 'channel_names': cand, 'source':'brainflow_csv'}
    return X, fs, meta

def read_openbci_any(path: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    p = path.lower()
    if p.endswith('.txt'):
        return _read_openbci_txt(path)
    elif p.endswith('.csv'):
        return _read_brainflow_csv(path)
    else:
        raise ValueError("Unsupported OpenBCI file (use .txt from GUI RAW or .csv BrainFlow export).")

def _butter_bandpass(low, high, fs, order=4):
    ny = 0.5*fs
    b, a = butter(order, [low/ny, high/ny], btype='band')
    return b, a

def _apply_bandpass(x, fs, band):
    b, a = _butter_bandpass(band[0], band[1], fs, 4)
    return filtfilt(b, a, x, axis=-1)

def _apply_notch(x, fs, notch_hz=60.0, Q=30.0):
    if notch_hz <= 0: return x
    b, a = iirnotch(w0=notch_hz/(fs/2.0), Q=Q)
    return filtfilt(b, a, x, axis=-1)

def _car(x):
    """Common Average Reference per-sample across channels."""
    return x - x.mean(axis=0, keepdims=True)

def _interp_nans(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaNs per channel."""
    for c in range(x.shape[0]):
        v = x[c]
        idx = np.arange(v.size)
        mask = ~np.isfinite(v)
        if mask.any():
            v[mask] = np.interp(idx[mask], idx[~mask], v[~mask])
        x[c] = v
    return x

def clean_openbci_signals(X: np.ndarray, fs: float, band=(0.5, 40.0), notch_hz=60.0,
                          saturations=OPENBCI_SAT_VALS, clip_uV=5000.0, use_car=True):
    """Clean 16-ch OpenBCI EEG.
    Steps:
      - Replace saturations and extreme outliers (|x| > clip_uV * 1e-6?) with NaN, then interpolate
      - Band-pass 0.5–40 Hz
      - Notch 50/60 Hz
      - Optional CAR re-referencing
      - z-score per channel
    NOTE: OpenBCI EXG values may be in microvolts depending on GUI export; the absolute scale
    is not critical for CNNs; the z-score removes scale differences.
    """
    X = X.copy().astype(np.float64)
    # mark saturations and absurd outliers as NaN
    bad = np.zeros_like(X, dtype=bool)
    for sat in saturations:
        bad |= (X == sat)
    # also mark super large spikes as NaN (>|clip| * 10 to be safe if unit differs)
    bad |= (np.abs(X) > (clip_uV * 10))
    X[bad] = np.nan
    X = _interp_nans(X)
    # detrend by removing per-channel mean
    X = X - X.mean(axis=1, keepdims=True)
    # filters
    X = _apply_bandpass(X, fs, band)
    X = _apply_notch(X, fs, notch_hz)
    if use_car and X.shape[0] >= 2:
        X = _car(X)
    # z-score per channel
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    return X

def resample_to(X: np.ndarray, fs_in: float, fs_out: float) -> Tuple[np.ndarray, float]:
    if abs(fs_in - fs_out) < 1e-6:
        return X, fs_in
    n_new = int(round(X.shape[1] * fs_out / fs_in))
    Y = resample(X, n_new, axis=1)
    return Y, fs_out

def slice_windows(X: np.ndarray, fs: float, win_sec: float, hop_sec: Optional[float]=None) -> np.ndarray:
    hop_sec = hop_sec if hop_sec is not None else win_sec
    w = int(round(win_sec * fs))
    h = int(round(hop_sec * fs))
    n = (X.shape[1] - w) // h + 1
    out = []
    for i in range(max(0, n)):
        seg = X[:, i*h:i*h+w]
        out.append(seg.astype(np.float32))
    if not out:
        return np.zeros((0, X.shape[0], int(win_sec*fs)), dtype=np.float32)
    return np.stack(out, axis=0)

def prepare_openbci_npz(in_path: str, out_npz: str,
                        task: str = "sleep",
                        sleep_fs=100.0, sleep_win=30.0,
                        seiz_fs=200.0, seiz_win=2.0, seiz_hop=0.5,
                        band=(0.5, 40.0), notch_hz=60.0, use_car=True) -> Dict[str, Any]:
    """Convert a single OpenBCI recording to model-ready NPZ for either 'sleep' or 'seizure' inference.
    """
    X, fs, meta = read_openbci_any(in_path)
    X = clean_openbci_signals(X, fs, band=band, notch_hz=notch_hz, use_car=use_car)
    if task == "sleep":
        Xr, fr = resample_to(X, fs, sleep_fs)
        W = slice_windows(Xr, fr, win_sec=sleep_win)
    else:
        Xr, fr = resample_to(X, fs, seiz_fs)
        W = slice_windows(Xr, fr, win_sec=seiz_win, hop_sec=seiz_hop)
    np.savez_compressed(out_npz, X=W, fs=fr, meta=meta, channels=meta.get('channel_names'))
    return {'n_windows': int(W.shape[0]), 'fs_out': fr, 'n_ch': int(W.shape[1])}
