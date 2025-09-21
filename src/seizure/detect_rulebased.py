import os, yaml, numpy as np
from scipy.signal import welch

def energy(x):
    return np.mean(x**2)

def band_ratio_3_4(x, fs):
    f, Pxx = welch(x, fs=fs, nperseg=min(1024,len(x)))
    mask = (f>=3) & (f<=4)
    num = np.trapz(Pxx[mask], f[mask])
    den = np.trapz(Pxx, f) + 1e-12
    return num/den

if __name__ == "__main__":
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    pdir = cfg["sleep_edf"]["processed_dir"]
    npz_path = os.path.join(pdir, "synthetic_seizure", "synthetic_seizure.npz")
    if not os.path.exists(npz_path):
        raise SystemExit("Run simulate.py first.")
    data = np.load(npz_path, allow_pickle=True)
    X, y, fs = data["X"], data["y"], int(data["fs"])

    # thresholds
    win = int(cfg["seizure"]["rule_based"]["energy_win_sec"] * fs)
    en_t = cfg["seizure"]["rule_based"]["energy_thresh_std"]
    r_t = cfg["seizure"]["rule_based"]["band_3_4_ratio_thresh"]

    # compute baseline energy stats on non-seizure windows
    en = np.array([energy(x) for x in X[y==0]])
    mu, sd = en.mean(), en.std()+1e-9
    th = mu + en_t*sd

    yp = []
    for x in X:
        # simple decision: high energy or high 3-4 Hz ratio
        yp.append( int( (energy(x) > th) or (band_ratio_3_4(x, fs) > r_t) ) )
    yp = np.array(yp)
    tp = int(((yp==1)&(y==1)).sum())
    fn = int(((yp==0)&(y==1)).sum())
    fp = int(((yp==1)&(y==0)).sum())
    tn = int(((yp==0)&(y==0)).sum())
    sens = tp/(tp+fn+1e-9)
    fpr = fp/(fp+tn+1e-9)
    print(f"Sensitivity: {sens*100:.1f}%")
    print(f"False positive rate (per window): {fpr*100:.1f}%")
