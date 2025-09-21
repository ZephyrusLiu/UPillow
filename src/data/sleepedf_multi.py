import os
from typing import Dict, List

import mne
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample

def _pick_eeg_channels(raw, prefs: List[str], max_ch: int):
    chosen = []
    chs = raw.ch_names
    # prioritize by prefs occurrences
    for p in prefs:
        for c in chs:
            if p.lower() in c.lower() and c not in chosen:
                chosen.append(c)
                if len(chosen) >= max_ch:
                    return chosen
    # fallback: any EEG channels
    for c in chs:
        if 'eeg' in c.lower() and c not in chosen:
            chosen.append(c)
            if len(chosen) >= max_ch:
                break
    if not chosen:
        chosen = [chs[0]]
    return chosen[:max_ch]

def _bandpass_notch(x, fs, band, notch_hz=None):
    ny = 0.5 * fs
    b, a = butter(4, [band[0]/ny, band[1]/ny], btype='band')
    y = filtfilt(b, a, x, axis=-1)
    if notch_hz and notch_hz > 0:
        q = 30.0
        b2, a2 = iirnotch(w0=notch_hz/(fs/2.0), Q=q)
        y = filtfilt(b2, a2, y, axis=-1)
    return y

def _map_stage(label: str, label_map: Dict[str, List[str]], classes: List[str]):
    L = label.strip().upper()
    for k, vals in label_map.items():
        if k not in classes:
            continue
        for v in vals:
            if v.upper() in L:
                return k
    if L.startswith("SLEEP STAGE "):
        L = L.replace("SLEEP STAGE ", "")
    if L in ["0", "W", "WAKE"]:
        return "W"
    if L in ["1", "N1"]:
        return "N1"
    if L in ["2", "N2"]:
        return "N2"
    if L in ["3", "4", "N3"]:
        return "N3"
    if L in ["R", "REM"]:
        return "REM"
    return None


def extract_epochs_multi(eeg_path: str, hyp_path: str, cfg: Dict, sid: str):
    sr_target = cfg["sleep_edf"]["sample_rate"]
    band = cfg["sleep_edf"]["bandpass"]
    notch = cfg["sleep_edf"]["notch_hz"]
    epoch_sec = cfg["sleep_edf"]["epoch_sec"]
    prefs = cfg["sleep_edf"].get("eeg_channel_preference_multi", ["Fpz-Cz","Pz-Oz","C3-A2","EEG"])
    n_ch = cfg["sleep_edf"].get("n_channels", 2)
    label_map = cfg["sleep_edf"]["label_map"]
    classes = ["W", "N1", "N2", "N3", "REM"]

    raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
    ch_sel = _pick_eeg_channels(raw, prefs, n_ch)
    raw.pick_channels(ch_sel)
    fs = raw.info["sfreq"]
    Xraw = raw.get_data()  # (C, T)

    # resample
    if abs(fs - sr_target) > 1e-3:
        n_new = int(Xraw.shape[1] * sr_target / fs)
        Xraw = resample(Xraw, n_new, axis=1)
        fs = sr_target

    Xraw = _bandpass_notch(Xraw, fs, band, notch)

    # annotations
    if hyp_path and os.path.exists(hyp_path):
        ann = mne.read_annotations(hyp_path, verbose=False)
        raw.set_annotations(ann, emit_warning=False)
        ann = raw.annotations
    else:
        ann = raw.annotations

    epoch_len = int(epoch_sec * fs)
    n_epochs = Xraw.shape[1] // epoch_len
    if n_epochs == 0:
        return (
            np.zeros((0, n_ch, epoch_len), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            classes,
            ch_sel,
            int(fs),
        )

    labels = np.full(n_epochs, -1, dtype=np.int32)
    if len(ann):
        onsets = np.asarray(ann.onset)
        durations = np.asarray(ann.duration)
        descriptions = np.asarray(ann.description)
        ends = onsets + durations
        for onset, end, desc in zip(onsets, ends, descriptions):
            mapped = _map_stage(desc, label_map, classes)
            if mapped is None:
                continue
            start_epoch = int(np.floor(onset / epoch_sec))
            stop_epoch = int(np.ceil(end / epoch_sec))
            if stop_epoch <= 0 or start_epoch >= n_epochs:
                continue
            start_epoch = max(start_epoch, 0)
            stop_epoch = min(stop_epoch, n_epochs)
            labels[start_epoch:stop_epoch] = classes.index(mapped)

    valid_idx = np.where(labels >= 0)[0]
    X = []
    y = []
    for idx in valid_idx:
        seg = Xraw[:, idx * epoch_len : (idx + 1) * epoch_len]
        seg = (seg - seg.mean(axis=1, keepdims=True)) / (
            seg.std(axis=1, keepdims=True) + 1e-8
        )
        X.append(seg.astype(np.float32))
        y.append(labels[idx])

    if X:
        X = np.stack(X)
        y = np.asarray(y, dtype=np.int64)
    else:
        X = np.zeros((0, n_ch, epoch_len), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
    return X, y, classes, ch_sel, int(fs)
