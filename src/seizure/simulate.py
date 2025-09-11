import os, yaml, numpy as np
from scipy.signal import chirp
from tqdm import tqdm

def inject_spike_wave(x, fs, hz=3.0, amp=4.0, dur=5.0):
    """Add a 3 Hz spike-and-wave like burst to center of the signal."""
    n = len(x)
    t = np.arange(n)/fs
    # center burst
    start = n//2 - int(dur*fs/2)
    end = start + int(dur*fs)
    burst = np.zeros(n, dtype=np.float32)
    tt = np.arange(end-start)/fs
    burst_seg = amp * (np.sign(np.sin(2*np.pi*hz*tt)) * np.exp(-((tt-dur/2)/(dur/6))**2))
    burst[start:end] = burst_seg[:end-start]
    return x + burst

if __name__ == "__main__":
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    pdir = cfg["sleep_edf"]["processed_dir"]
    out = os.path.join(pdir, "synthetic_seizure")
    os.makedirs(out, exist_ok=True)
    sr = cfg["sleep_edf"]["sample_rate"]
    prob = cfg["seizure"]["synthetic"]["prob_seizure_window"]
    hz = cfg["seizure"]["synthetic"]["burst_hz"]
    amp = cfg["seizure"]["synthetic"]["burst_amp"]
    dur = cfg["seizure"]["synthetic"]["burst_sec"]

    # concatenate all clean windows and randomly inject seizures
    X_all = []; y_all = []
    import glob, numpy as np
    for f in glob.glob(os.path.join(pdir, "*.npz")):
        if "manifest" in f: continue
        npz = np.load(f, allow_pickle=True)
        X = npz["X"]  # (N,T)
        # create labels: 0 = non-seizure, 1 = seizure (synthetic)
        for x in X:
            if np.random.rand() < prob:
                xs = inject_spike_wave(x.copy(), sr, hz=hz, amp=amp, dur=dur)
                X_all.append(xs); y_all.append(1)
            else:
                X_all.append(x); y_all.append(0)
    X_all = np.stack(X_all).astype("float32")
    y_all = np.array(y_all, dtype="int64")
    np.savez_compressed(os.path.join(out, "synthetic_seizure.npz"), X=X_all, y=y_all, fs=sr)
    print(f"Saved {len(y_all)} windows to {out}/synthetic_seizure.npz; seizure ratio={y_all.mean():.3f}")
