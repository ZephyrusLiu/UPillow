import os, re, csv, json
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample
from typing import Tuple, Dict, Any, Optional, List, Sequence

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
                          saturations=OPENBCI_SAT_VALS, clip_uV=5000.0, use_car=True,
                          return_pre_zscore: bool = False):
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
    pre_z = X.copy()
    # z-score per channel
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    if return_pre_zscore:
        return X, pre_z
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

def _resolve_indices(names: Sequence[str], items: Sequence[Any]) -> List[int]:
    """Resolve a list of channel identifiers (name or index) to index positions."""
    idx = {str(n): i for i, n in enumerate(names)}
    resolved: List[int] = []
    for it in items:
        if isinstance(it, (int, np.integer)):
            if 0 <= int(it) < len(names):
                resolved.append(int(it))
        else:
            key = str(it)
            if key in idx:
                resolved.append(idx[key])
    return resolved


def build_virtual_channels(X: np.ndarray,
                           channel_names: Sequence[str],
                           virtual_map: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Construct virtual bipolar channels via quality-weighted averaging.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (C, T) containing cleaned (pre z-score) signals.
    channel_names : list-like
        Names corresponding to each row of ``X``.
    virtual_map : sequence of dicts
        Each dict should contain ``pos`` and ``neg`` lists specifying which
        channels contribute to the positive and negative poles. Optional keys:
        ``name`` (virtual channel label) and ``min_channels`` (minimum number of
        available contributors before the channel is considered valid).

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        The stacked virtual channels (V, T) and a metadata dictionary capturing
        the weights used for each contributor.
    """
    if not virtual_map:
        raise ValueError("virtual_map must contain at least one channel definition")

    ch_names = list(channel_names)
    quality = np.nanstd(X, axis=1)
    quality = np.where(np.isfinite(quality), quality, 0.0)

    virtual_signals: List[np.ndarray] = []
    weight_meta: Dict[str, Any] = {}

    for i, spec in enumerate(virtual_map):
        pos_items = spec.get('pos', [])
        neg_items = spec.get('neg', [])
        pos_idx = _resolve_indices(ch_names, pos_items)
        neg_idx = _resolve_indices(ch_names, neg_items)
        min_required = int(spec.get('min_channels', 1))
        if len(pos_idx) < min_required or len(neg_idx) < min_required:
            raise ValueError(f"Virtual channel {spec.get('name', i)} missing required contributors")

        pos_q = quality[pos_idx]
        neg_q = quality[neg_idx]
        if pos_q.sum() <= 0:
            pos_w = np.ones(len(pos_idx)) / len(pos_idx)
        else:
            pos_w = pos_q / pos_q.sum()
        if neg_q.sum() <= 0:
            neg_w = np.ones(len(neg_idx)) / len(neg_idx)
        else:
            neg_w = neg_q / neg_q.sum()

        pos_sig = np.sum(X[pos_idx] * pos_w[:, None], axis=0)
        neg_sig = np.sum(X[neg_idx] * neg_w[:, None], axis=0)
        virt = pos_sig - neg_sig
        # final z-score for stability
        virt = virt - np.mean(virt)
        std = np.std(virt)
        virt = virt / (std + 1e-8)
        virtual_signals.append(virt.astype(np.float32))

        label = spec.get('name', f'virtual_{i}')
        weight_meta[label] = {
            'pos': {ch_names[j]: float(pos_w[k]) for k, j in enumerate(pos_idx)},
            'neg': {ch_names[j]: float(neg_w[k]) for k, j in enumerate(neg_idx)}
        }

    V = np.stack(virtual_signals, axis=0)
    return V, weight_meta


def prepare_openbci_npz(in_path: str, out_npz: str,
                        task: str = "sleep",
                        sleep_fs=100.0, sleep_win=30.0,
                        seiz_fs=200.0, seiz_win=2.0, seiz_hop=0.5,
                        band=(0.5, 40.0), notch_hz=60.0, use_car=True,
                        virtual_map: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Convert a single OpenBCI recording to model-ready NPZ for either 'sleep' or 'seizure' inference.

    When ``virtual_map`` is supplied, each entry defines a bipolar combination of
    channels ("pos" minus "neg") so that multi-channel pillow arrays can be
    projected onto two-channel layouts compatible with pretrained Sleep-EDF
    models. Channel identifiers may be integers (indices) or strings matching the
    names parsed from the OpenBCI header.
    """
    X, fs, meta = read_openbci_any(in_path)
    if virtual_map:
        _, pre_z = clean_openbci_signals(X, fs, band=band, notch_hz=notch_hz,
                                         use_car=use_car, return_pre_zscore=True)
        virt, weight_meta = build_virtual_channels(pre_z, meta.get('channel_names', []), virtual_map)
        X_proc = virt
        meta = meta.copy()
        meta['virtual_map'] = weight_meta
        meta['source_channel_names'] = meta.get('channel_names', [])
        meta['channel_names'] = [spec.get('name', f'virtual_{i}') for i, spec in enumerate(virtual_map)]
    else:
        X_proc = clean_openbci_signals(X, fs, band=band, notch_hz=notch_hz, use_car=use_car)
    base = X_proc
    if task == "sleep":
        Xr, fr = resample_to(base, fs, sleep_fs)
        W = slice_windows(Xr, fr, win_sec=sleep_win)
    else:
        Xr, fr = resample_to(base, fs, seiz_fs)
        W = slice_windows(Xr, fr, win_sec=seiz_win, hop_sec=seiz_hop)
    np.savez_compressed(out_npz, X=W, fs=fr, meta=meta, channels=meta.get('channel_names'))
    return {'n_windows': int(W.shape[0]), 'fs_out': fr, 'n_ch': int(W.shape[1])}
