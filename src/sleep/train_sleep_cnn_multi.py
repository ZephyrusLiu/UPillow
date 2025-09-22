import os, glob, yaml, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from src.models.cnn_1d_multi import TinySleepCNNMulti
from src.data.sleepedf_multi import extract_epochs_multi
import mne

class NPZMulti(Dataset):
    def __init__(self, X, y):
        self.X = X.astype("float32")
        self.y = y.astype("int64")
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        import torch
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

def load_sleepedf_multi(processed_dir):
    files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
    data = []
    for f in files:
        if "manifest" in f: continue
        npz = np.load(f, allow_pickle=True)
        X = npz["X"]; y = npz["y"]
        if X.ndim == 2:  # single-channel fallback
            X = X[:,None,:]
        data.append((os.path.basename(f), X, y))
    return data

if __name__ == "__main__":
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    pdir = cfg["sleep_edf"]["processed_dir"]
    data = load_sleepedf_multi(pdir)
    if not data:
        raise SystemExit("Please run sleepedf preprocessing (single or multi) first.")
    # subject-wise split by file
    rng = np.random.default_rng(cfg["training"]["random_state"])
    idx = np.arange(len(data)); rng.shuffle(idx)
    n_te = max(1, int(len(idx)*cfg["training"]["test_subject_frac"]))
    te_idx = set(idx[:n_te])
    Xtr = np.concatenate([data[i][1] for i in range(len(data)) if i not in te_idx], axis=0)
    ytr = np.concatenate([data[i][2] for i in range(len(data)) if i not in te_idx], axis=0)
    Xte = np.concatenate([data[i][1] for i in range(len(data)) if i in te_idx], axis=0)
    yte = np.concatenate([data[i][2] for i in range(len(data)) if i in te_idx], axis=0)

    in_len = Xtr.shape[-1]; in_ch = Xtr.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinySleepCNNMulti(n_classes=5, in_len=in_len, in_ch=in_ch).to(device)
    lr = float(cfg["training"]["lr"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()

    dl_tr = DataLoader(NPZMulti(Xtr,ytr), batch_size=cfg["training"]["batch_size"], shuffle=True)
    dl_te = DataLoader(NPZMulti(Xte,yte), batch_size=cfg["training"]["batch_size"], shuffle=False)

    best = 0.0
    for ep in range(cfg["training"]["epochs"]):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        # eval
        model.eval()
        yp_all, yt_all = [], []
        with torch.no_grad():
            for xb, yb in dl_te:
                xb = xb.to(device)
                yp = model(xb).argmax(1).cpu().numpy()
                yp_all.append(yp); yt_all.append(yb.numpy())
        yp_all = np.concatenate(yp_all); yt_all = np.concatenate(yt_all)
        acc = accuracy_score(yt_all, yp_all)
        f1 = f1_score(yt_all, yp_all, average="macro")
        kap = cohen_kappa_score(yt_all, yp_all)
        print(f"Epoch {ep+1}: acc={acc:.3f}, f1={f1:.3f}, kappa={kap:.3f}")
        if acc > best:
            best = acc
            os.makedirs("models/ckpts", exist_ok=True)
            torch.save(model.state_dict(), "models/ckpts/sleep_cnn_multi_best.pt")
    print(f"Best acc: {best:.3f}")
