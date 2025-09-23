import yaml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src.features.spectral import band_powers
from src.sleep.data_utils import load_sleepedf_multi_npz


def make_multi_features(X: np.ndarray, fs: int) -> np.ndarray:
    if fs is None:
        raise ValueError("Sampling rate (fs) is required to compute spectral features.")
    if X.ndim == 2:
        X = X[:, None, :]
    feats = []
    for epoch in X:
        chan_feats = [band_powers(epoch[ch], fs) for ch in range(epoch.shape[0])]
        feats.append(np.concatenate(chan_feats, axis=0))
    return np.stack(feats)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["sleep_edf"]["processed_dir"]
    data = load_sleepedf_multi_npz(processed_dir)
    if not data:
        raise SystemExit("No processed Sleep-EDF files found. Run the preprocessing scripts first.")

    accs, f1s, kaps = [], [], []
    for i, (sid, X, y, fs) in enumerate(data):
        Xtr, ytr = [], []
        for j, (sid2, X2, y2, fs2) in enumerate(data):
            if i == j:
                continue
            Xtr.append(make_multi_features(X2, fs2))
            ytr.append(y2)
        if not Xtr:
            continue
        Xtr = np.concatenate(Xtr, axis=0)
        ytr = np.concatenate(ytr, axis=0)
        Xte = make_multi_features(X, fs)
        yte = y

        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=cfg["training"]["random_state"],
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        accs.append(accuracy_score(yte, yp))
        f1s.append(f1_score(yte, yp, average="macro"))
        kaps.append(cohen_kappa_score(yte, yp))
        print(f"[{sid}] acc={accs[-1]:.3f}, f1={f1s[-1]:.3f}, kappa={kaps[-1]:.3f}")

    if accs:
        print(
            f"LOSO mean: acc={np.mean(accs):.3f}±{np.std(accs):.3f}, "
            f"f1={np.mean(f1s):.3f}, kappa={np.mean(kaps):.3f}"
        )
    else:
        print("Insufficient subjects for cross-validation.")
