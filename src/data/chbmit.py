import os, re, glob
from typing import Dict, List, Tuple
import numpy as np
import mne
from scipy.signal import butter, iirnotch, filtfilt, resample

def _pick_eeg_channel(raw: mne.io.BaseRaw, prefs: List[str]) -> str:
    chs = raw.ch_names
    for p in prefs:
        for c in chs:
            if p.lower() in c.lower():
                return c
    for c in chs:
        if "eeg" in c.lower():
            return c
    return chs[0]

def _bandpass_notch(x, fs, band, notch_hz=None):
    ny = 0.5 * fs
    b, a = butter(4, [band[0]/ny, band[1]/ny], btype='band')
    y = filtfilt(b, a, x)
    if notch_hz and notch_hz > 0:
        q = 30.0
        b2, a2 = iirnotch(w0=notch_hz/(fs/2.0), Q=q)
        y = filtfilt(b2, a2, y)
    return y

def parse_summaries(root: str) -> Dict[str, List[Tuple[float, float]]]:
    """Parse *-summary.txt files under CHB-MIT root to get seizure [start, end] (sec) per EDF.
    Returns mapping: edf_basename -> list of (start_sec, end_sec)
    """
    mapping: Dict[str, List[Tuple[float, float]]] = {}
    summ_files = glob.glob(os.path.join(root, "**", "*summary*.txt"), recursive=True)
    # Some distributions use chbXX-summary.txt
    if not summ_files:
        summ_files = glob.glob(os.path.join(root, "**", "chb*-summary.txt"), recursive=True)
    for sf in summ_files:
        with open(sf, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f.readlines()]
        cur_file = None
        for ln in lines:
            mfile = re.search(r"File Name:\s*([A-Za-z0-9_]+\.edf)", ln, re.I)
            if mfile:
                cur_file = mfile.group(1)
                mapping.setdefault(cur_file, [])
                continue
            ms = re.search(r"Seizure Start Time:\s*([0-9]+)\s*seconds", ln, re.I)
            me = re.search(r"Seizure End Time:\s*([0-9]+)\s*seconds", ln, re.I)
            if ms:
                start = float(ms.group(1))
                # lookahead for end on same or next lines
                # we will collect when end appears
                last_start = start
            if me:
                end = float(me.group(1))
                # use last_start if exists
                try:
                    mapping[cur_file].append((last_start, end))
                except Exception:
                    # If file had end before start (rare formatting), skip
                    pass
    return mapping

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def edf_to_windows(edf_path: str, seizures: List[Tuple[float, float]], cfg: dict):
    sr_t = cfg["chb_mit"]["sample_rate"]
    band = cfg["chb_mit"]["bandpass"]
    notch = cfg["chb_mit"]["notch_hz"]
    wsec = cfg["chb_mit"]["window_sec"]
    ovlp = cfg["chb_mit"]["overlap_sec"]
    prefs = cfg["chb_mit"]["eeg_channel_preference"]

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    ch = _pick_eeg_channel(raw, prefs)
    raw.pick_channels([ch])
    fs0 = raw.info["sfreq"]
    x = raw.get_data()[0]
    # Resample
    if abs(fs0 - sr_t) > 1e-3:
        n_new = int(len(x) * sr_t / fs0)
        x = resample(x, n_new)
        fs = sr_t
    else:
        fs = fs0
    # Filter & z-score per 10s block to avoid drift
    x = _bandpass_notch(x, fs, band, notch)
    # windows
    wlen = int(wsec * fs)
    n = len(x) // wlen
    X = []
    y = []
    for i in range(n):
        t0 = i * wsec
        t1 = t0 + wsec
        lab = 0
        for (s0, s1) in seizures:
            if _overlap(t0, t1, s0, s1) >= ovlp:
                lab = 1
                break
        seg = x[i*wlen:(i+1)*wlen]
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        X.append(seg.astype(np.float32))
        y.append(lab)
    return np.stack(X), np.array(y, dtype=np.int64), ch, int(fs)

def build_index(root: str) -> Dict[str, str]:
    """Map edf basename -> full path (search recursively)."""
    idx = {}
    for p in glob.glob(os.path.join(root, "**", "*.edf"), recursive=True):
        idx[os.path.basename(p)] = p
    return idx

def prepare_chbmit(cfg: dict):
    raw_dir = cfg["chb_mit"]["raw_dir"]
    out_dir = cfg["chb_mit"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)

    summaries = parse_summaries(raw_dir)
    edf_index = build_index(raw_dir)
    manifest = []
    for edf_name, seiz_list in summaries.items():
        if edf_name not in edf_index:
            print(f"[WARN] Missing EDF for {edf_name}")
            continue
        path = edf_index[edf_name]
        try:
            X, y, ch, fs = edf_to_windows(path, seiz_list, cfg)
            if len(X) == 0:
                continue
            base = os.path.splitext(edf_name)[0]
            npz_path = os.path.join(out_dir, f"{base}.npz")
            np.savez_compressed(npz_path, X=X, y=y, ch=ch, fs=fs, edf=edf_name)
            # patient id e.g., chb01
            m = re.match(r"(chb\d+)_", base, re.I)
            pid = m.group(1).lower() if m else "unknown"
            manifest.append({"edf": edf_name, "n": int(len(X)), "pos": int(y.sum()), "ch": ch, "fs": fs, "patient": pid})
            print(f"[OK] {edf_name}: {len(X)} windows ({y.sum()} sz), ch={ch}, fs={fs}")
        except Exception as e:
            print(f"[WARN] {edf_name} failed: {e}")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        import json
        json.dump(manifest, f, indent=2)
    return manifest
