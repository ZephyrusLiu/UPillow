import os
import re
import glob
import numpy as np
import mne
from scipy.signal import iirnotch, filtfilt, butter, resample
from typing import Dict, List, Tuple

def _pick_eeg_channel(raw: mne.io.BaseRaw, prefs: List[str]) -> str:
    chs = raw.ch_names
    for p in prefs:
        for c in chs:
            if p.lower() in c.lower():
                return c
    # fallback: first EEG channel
    for c in chs:
        if 'eeg' in c.lower():
            return c
    return chs[0]

def _bandpass_notch(x, fs, band, notch_hz=None):
    # bandpass
    ny = 0.5 * fs
    b, a = butter(4, [band[0]/ny, band[1]/ny], btype='band')
    y = filtfilt(b, a, x)
    # notch
    if notch_hz is not None and notch_hz > 0:
        q = 30.0
        b2, a2 = iirnotch(w0=notch_hz/(fs/2.0), Q=q)
        y = filtfilt(b2, a2, y)
    return y

def _map_stage(label: str, label_map: Dict[str, List[str]]):
    L = label.strip().upper()
    for k, vals in label_map.items():
        for v in vals:
            if v.upper() in L:
                return k
    # handle "Sleep stage X" formats
    if L.startswith("SLEEP STAGE "):
        t = L.replace("SLEEP STAGE ", "")
        if t in ['0', 'W']:
            return 'W'
        if t in ['1', 'N1']:
            return 'N1'
        if t in ['2', 'N2']:
            return 'N2'
        if t in ['3', '4', 'N3']:
            return 'N3'
        if t in ['R', 'REM']:
            return 'REM'
    return None

def load_sleepedf_subjects(raw_dir: str) -> Dict[str, Dict[str, str]]:
    """Return mapping subject_id -> { 'eeg': path, 'hyp': path } for Sleep-EDF.
    This handles typical naming but you may need to adapt for your folder layout.
    """
    pairs = {}
    edfs = glob.glob(os.path.join(raw_dir, "**", "*.edf"), recursive=True)
    # naive pairing by filename stems
    for e in edfs:
        name = os.path.basename(e)
        if "Hypnogram" in name or "hypnogram" in name:
            continue
        stem = re.sub(r"\.edf$", "", name, flags=re.I)
        # try to find hypnogram in same folder
        hyp = None
        for h in edfs:
            if h == e: 
                continue
            hn = os.path.basename(h)
            if "Hypnogram" in hn and stem.split("-")[0] in hn:
                hyp = h
                break
        if hyp is None:
            # also consider hypnogram stored as annotations inside recording
            hyp = ""  # empty means: use annotations from same file if present
        sid = stem.split("-")[0]
        pairs[sid] = {"eeg": e, "hyp": hyp}
    return pairs

def extract_epochs(eeg_path: str, hyp_path: str, cfg: Dict, sid: str):
    sr_target = cfg["sleep_edf"]["sample_rate"]
    band = cfg["sleep_edf"]["bandpass"]
    notch = cfg["sleep_edf"]["notch_hz"]
    epoch_sec = cfg["sleep_edf"]["epoch_sec"]
    prefs = cfg["sleep_edf"]["eeg_channel_preference"]
    label_map = cfg["sleep_edf"]["label_map"]

    raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
    ch = _pick_eeg_channel(raw, prefs)
    raw.pick_channels([ch])
    fs = raw.info["sfreq"]

    x = raw.get_data()[0]
    # resample to target
    if abs(fs - sr_target) > 1e-3:
        n_new = int(len(x) * sr_target / fs)
        x = resample(x, n_new)
        fs = sr_target

    # filter
    x = _bandpass_notch(x, fs, band, notch)

    # annotations
    if hyp_path and os.path.exists(hyp_path):
        ann = mne.read_annotations(hyp_path, verbose=False)
    else:
        ann = raw.annotations

    # build epoch labels from annotations
    epoch_len = int(epoch_sec * fs)
    n_epochs = len(x) // epoch_len
    X = []
    y = []
    for i in range(n_epochs):
        start_t = i * epoch_sec
        # find annotation covering this time
        label = None
        for a in ann:
            if a["onset"] <= start_t < (a["onset"] + a["duration"]):
                label = _map_stage(a["description"], label_map)
                break
        if label is None:
            continue
        seg = x[i*epoch_len:(i+1)*epoch_len]
        # z-score per epoch
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        X.append(seg.astype(np.float32))
        y.append(label)
    X = np.stack(X) if X else np.zeros((0, epoch_len), dtype=np.float32)
    y = np.array(y)
    # map labels to ints
    classes = ["W","N1","N2","N3","REM"]
    y_idx = np.array([classes.index(s) for s in y if s in classes])
    # align y and X properly
    mask = np.array([s in classes for s in y])
    return X[mask], y_idx, classes, ch, fs

def save_processed(raw_dir: str, out_dir: str, cfg: Dict):
    os.makedirs(out_dir, exist_ok=True)
    pairs = load_sleepedf_subjects(raw_dir)
    manifest = []
    for sid, paths in pairs.items():
        try:
            X, y, classes, ch, fs = extract_epochs(paths["eeg"], paths["hyp"], cfg, sid)
            if len(X) == 0:
                continue
            np.savez_compressed(os.path.join(out_dir, f"{sid}.npz"), X=X, y=y, classes=classes, ch=ch, fs=fs)
            manifest.append({"sid": sid, "n": int(len(X)), "channel": ch, "fs": fs})
            print(f"[OK] {sid}: {len(X)} epochs, ch={ch}, fs={fs}")
        except Exception as e:
            print(f"[WARN] {sid} failed: {e}")
    # save manifest
    import json
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
