import os, glob, json, yaml, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report
from src.features.spectral import band_powers
from tqdm import tqdm

def load_all_npz(processed_dir):
    files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
    data = []
    for f in files:
        if os.path.basename(f).startswith("manifest"):
            continue
        npz = np.load(f, allow_pickle=True)
        X = npz["X"]
        y = npz["y"]
        fs = int(npz["fs"])
        sid = os.path.basename(f).split(".")[0]
        data.append((sid, X, y, fs))
    return data

def make_features(X, fs):
    feats = [band_powers(x, fs) for x in X]
    return np.stack(feats)

if __name__ == "__main__":
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    pdir = cfg["sleep_edf"]["processed_dir"]
    rng = np.random.default_rng(cfg["training"]["random_state"])
    data = load_all_npz(pdir)
    if not data:
        raise SystemExit("No processed files found. Run prepare_sleepedf.py first.")
    # Leave-one-subject-out
    all_sids = [sid for sid,_,_,_ in data]
    accs, f1s, kaps = [], [], []
    for i, (sid, X, y, fs) in enumerate(data):
        Xtr, ytr = [], []
        for j, (sid2, X2, y2, fs2) in enumerate(data):
            if i == j: continue
            Xtr.append(make_features(X2, fs2))
            ytr.append(y2)
        Xtr = np.concatenate(Xtr, axis=0)
        ytr = np.concatenate(ytr, axis=0)
        Xte = make_features(X, fs)
        yte = y
        clf = RandomForestClassifier(n_estimators=300, random_state=cfg["training"]["random_state"], n_jobs=-1)
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        accs.append(accuracy_score(yte, yp))
        f1s.append(f1_score(yte, yp, average="macro"))
        kaps.append(cohen_kappa_score(yte, yp))
        print(f"[{sid}] acc={accs[-1]:.3f}, f1={f1s[-1]:.3f}, kappa={kaps[-1]:.3f}")
    print(f"LOSO mean: acc={np.mean(accs):.3f}±{np.std(accs):.3f}, f1={np.mean(f1s):.3f}, kappa={np.mean(kaps):.3f}")
