import os, glob
import numpy as np
import mne
from scipy.signal import butter, iirnotch, filtfilt, resample
from typing import List, Dict

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

def extract_epochs_multi(eeg_path: str, hyp_path: str, cfg: Dict, sid: str):
    sr_target = cfg["sleep_edf"]["sample_rate"]
    band = cfg["sleep_edf"]["bandpass"]
    notch = cfg["sleep_edf"]["notch_hz"]
    epoch_sec = cfg["sleep_edf"]["epoch_sec"]
    prefs = cfg["sleep_edf"].get("eeg_channel_preference_multi", ["Fpz-Cz","Pz-Oz","C3-A2","EEG"])
    n_ch = cfg["sleep_edf"].get("n_channels", 2)
    label_map = cfg["sleep_edf"]["label_map"]
    classes = ["W","N1","N2","N3","REM"]

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
    else:
        ann = raw.annotations

    epoch_len = int(epoch_sec * fs)
    n_epochs = Xraw.shape[1] // epoch_len
    X = []
    y = []
    def _map_stage(L: str):
        s = L.strip().upper()
        if s.startswith("SLEEP STAGE "):
            s = s.replace("SLEEP STAGE ", "")
        if s in ["0","W","WAKE"]: return "W"
        if s in ["1","N1"]: return "N1"
        if s in ["2","N2"]: return "N2"
        if s in ["3","4","N3"]: return "N3"
        if s in ["R","REM"]: return "REM"
        return None

    for i in range(n_epochs):
        start_t = i * epoch_sec
        label = None
        for a in ann:
            if a["onset"] <= start_t < (a["onset"] + a["duration"]):
                label = _map_stage(a["description"])
                break
        if label is None or label not in classes:
            continue
        seg = Xraw[:, i*epoch_len:(i+1)*epoch_len]
        # z-score per channel per epoch
        seg = (seg - seg.mean(axis=1, keepdims=True)) / (seg.std(axis=1, keepdims=True) + 1e-8)
        X.append(seg.astype(np.float32))
        y.append(classes.index(label))
    X = np.stack(X) if X else np.zeros((0, n_ch, epoch_len), dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, classes, ch_sel, int(fs)
