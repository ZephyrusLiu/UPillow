import os, glob, re
import numpy as np
import mne
from scipy.signal import butter, iirnotch, filtfilt, resample
from typing import List, Dict, Tuple

def _pick_eeg_channels(raw, prefs: List[str], max_ch: int):
    chosen = []
    chs = raw.ch_names
    for p in prefs:
        for c in chs:
            if p.lower() in c.lower() and c not in chosen:
                chosen.append(c)
                if len(chosen) >= max_ch:
                    return chosen
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

def parse_summaries(root: str) -> Dict[str, List[Tuple[float, float]]]:
    mapping = {}
    summ_files = glob.glob(os.path.join(root, "**", "*summary*.txt"), recursive=True)
    if not summ_files:
        summ_files = glob.glob(os.path.join(root, "**", "chb*-summary.txt"), recursive=True)
    for sf in summ_files:
        with open(sf, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f.readlines()]
        cur_file = None; last_start = None
        for ln in lines:
            mfile = re.search(r"File Name:\s*([A-Za-z0-9_]+\.edf)", ln, re.I)
            if mfile:
                cur_file = mfile.group(1); mapping.setdefault(cur_file, []); last_start=None; continue
            ms = re.search(r"Seizure Start Time:\s*([0-9]+)\s*seconds", ln, re.I)
            me = re.search(r"Seizure End Time:\s*([0-9]+)\s*seconds", ln, re.I)
            if ms: last_start = float(ms.group(1))
            if me and last_start is not None:
                mapping[cur_file].append((last_start, float(me.group(1)))); last_start=None
    return mapping

def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))

def edf_to_windows_multi(edf_path: str, seizures: List[Tuple[float,float]], cfg: dict):
    sr_t = cfg["chb_mit"]["sample_rate"]
    band = cfg["chb_mit"]["bandpass"]
    notch = cfg["chb_mit"]["notch_hz"]
    wsec = cfg["chb_mit"]["window_sec"]
    ovlp = cfg["chb_mit"]["overlap_sec"]
    prefs = cfg["chb_mit"].get("eeg_channel_preference_multi", ["EEG","FP1-F7","FP1-F3","C3-P3","CZ-PZ"])
    n_ch = cfg["chb_mit"].get("n_channels", 3)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    ch_sel = _pick_eeg_channels(raw, prefs, n_ch)
    raw.pick_channels(ch_sel)
    fs0 = raw.info["sfreq"]
    Xraw = raw.get_data()  # (C, T)
    # resample
    if abs(fs0 - sr_t) > 1e-3:
        n_new = int(Xraw.shape[1] * sr_t / fs0)
        Xraw = resample(Xraw, n_new, axis=1)
        fs = sr_t
    else:
        fs = fs0
    Xraw = _bandpass_notch(Xraw, fs, band, notch)

    wlen = int(wsec * fs)
    n = Xraw.shape[1] // wlen
    X = []; y = []
    for i in range(n):
        t0 = i * wsec; t1 = t0 + wsec
        lab = 0
        for (s0, s1) in seizures:
            if _overlap(t0, t1, s0, s1) >= ovlp:
                lab = 1; break
        seg = Xraw[:, i*wlen:(i+1)*wlen]
        seg = (seg - seg.mean(axis=1, keepdims=True)) / (seg.std(axis=1, keepdims=True) + 1e-8)
        X.append(seg.astype(np.float32)); y.append(lab)
    return np.stack(X), np.array(y, dtype=np.int64), ch_sel, int(fs)
