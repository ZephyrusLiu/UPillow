import os, glob, yaml, re
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score

class NPZBin(Dataset):
    def __init__(self, X, y):
        self.X = X[:,None,:].astype("float32")
        self.y = y.astype("int64")
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        import torch
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

def load_by_patient(pdir):
    files = [f for f in glob.glob(os.path.join(pdir, "*.npz")) if "manifest" not in f]
    groups = {}
    for f in files:
        base = os.path.basename(f).replace(".npz","")
        m = re.match(r"(chb\d+)_", base, re.I)
        pid = m.group(1).lower() if m else "unknown"
        data = np.load(f, allow_pickle=True)
        X, y = data["X"], data["y"]
        groups.setdefault(pid, {"X": [], "y": []})
        groups[pid]["X"].append(X); groups[pid]["y"].append(y)
    for k in list(groups.keys()):
        groups[k]["X"] = np.concatenate(groups[k]["X"], axis=0)
        groups[k]["y"] = np.concatenate(groups[k]["y"], axis=0)
    return groups

def split_patients(groups, holdout_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    pids = sorted(groups.keys())
    rng.shuffle(pids)
    n_te = max(1, int(len(pids)*holdout_frac))
    te = set(pids[:n_te])
    tr = [pid for pid in pids if pid not in te]
    return tr, list(te)

if __name__ == "__main__":
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)

    pdir = cfg["chb_mit"]["processed_dir"]
    groups = load_by_patient(pdir)
    tr_pids, te_pids = split_patients(groups, cfg["chb_mit"]["patient_holdout_frac"], 42)

    Xtr = np.concatenate([groups[p]["X"] for p in tr_pids], axis=0)
    ytr = np.concatenate([groups[p]["y"] for p in tr_pids], axis=0)
    Xte = np.concatenate([groups[p]["X"] for p in te_pids], axis=0)
    yte = np.concatenate([groups[p]["y"] for p in te_pids], axis=0)

    from src.models.cnn_1d import TinySleepCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinySleepCNN(n_classes=2, in_len=Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()

    dl_tr = DataLoader(NPZBin(Xtr,ytr), batch_size=128, shuffle=True, drop_last=False)
    dl_te = DataLoader(NPZBin(Xte,yte), batch_size=256, shuffle=False, drop_last=False)

    best = 0.0
    for ep in range(12):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        # eval
        model.eval()
        yp_all, pr_all = [], []
        yt_all = []
        with torch.no_grad():
            for xb, yb in dl_te:
                xb = xb.to(device)
                logits = model(xb)
                prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
                yp = logits.argmax(1).cpu().numpy()
                yp_all.append(yp); pr_all.append(prob); yt_all.append(yb.numpy())
        import numpy as np
        yp_all = np.concatenate(yp_all); yt_all = np.concatenate(yt_all); pr_all = np.concatenate(pr_all)
        acc = accuracy_score(yt_all, yp_all)
        p, r, f1, _ = precision_recall_fscore_support(yt_all, yp_all, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(yt_all, pr_all)
        except Exception:
            auc = float("nan")
        print(f"Epoch {ep+1}: acc={acc:.3f}, F1={f1:.3f}, P={p:.3f}, R={r:.3f}, AUC={auc:.3f}")
        if f1 > best:
            best = f1
            os.makedirs("models/ckpts", exist_ok=True)
            torch.save(model.state_dict(), "models/ckpts/chbmit_cnn_best.pt")
    print("Holdout patients:", te_pids)
    print(f"Best F1: {best:.3f}")
