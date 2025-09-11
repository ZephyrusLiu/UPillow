import os, glob, yaml, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from src.models.cnn_1d import TinySleepCNN
from tqdm import tqdm

class NPZDataset(Dataset):
    def __init__(self, X, y):
        self.X = X[:,None,:].astype("float32")  # (N,1,T)
        self.y = y.astype("int64")
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

def load_all_npz(processed_dir):
    files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
    data = []
    for f in files:
        if os.path.basename(f).startswith("manifest"):
            continue
        npz = np.load(f, allow_pickle=True)
        X = npz["X"]
        y = npz["y"]
        data.append((os.path.basename(f), X, y))
    return data

def split_subjects(data, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    n_test = max(1, int(len(idx)*test_frac))
    te_idx = set(idx[:n_test])
    tr, te = [], []
    for i, item in enumerate(data):
        (te if i in te_idx else tr).append(item)
    return tr, te

if __name__ == "__main__":
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    pdir = cfg["sleep_edf"]["processed_dir"]
    data = load_all_npz(pdir)
    if not data:
        raise SystemExit("No processed files found. Run prepare_sleepedf.py first.")
    tr, te = split_subjects(data, cfg["training"]["test_subject_frac"], cfg["training"]["random_state"])
    Xtr = np.concatenate([d[1] for d in tr], axis=0); ytr = np.concatenate([d[2] for d in tr], axis=0)
    Xte = np.concatenate([d[1] for d in te], axis=0); yte = np.concatenate([d[2] for d in te], axis=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_len = Xtr.shape[1]
    model = TinySleepCNN(n_classes=5, in_len=in_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    crit = torch.nn.CrossEntropyLoss()

    dl_tr = DataLoader(NPZDataset(Xtr,ytr), batch_size=cfg["training"]["batch_size"], shuffle=True)
    dl_te = DataLoader(NPZDataset(Xte,yte), batch_size=cfg["training"]["batch_size"], shuffle=False)

    best_acc = 0.0
    for ep in range(cfg["training"]["epochs"]):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        # eval
        model.eval()
        yp_all, yt_all = [], []
        with torch.no_grad():
            for xb, yb in dl_te:
                xb = xb.to(device)
                logits = model(xb)
                yp = logits.argmax(1).cpu().numpy()
                yp_all.append(yp)
                yt_all.append(yb.numpy())
        yp_all = np.concatenate(yp_all); yt_all = np.concatenate(yt_all)
        acc = accuracy_score(yt_all, yp_all)
        f1 = f1_score(yt_all, yp_all, average="macro")
        kap = cohen_kappa_score(yt_all, yp_all)
        print(f"Epoch {ep+1}: acc={acc:.3f}, f1={f1:.3f}, kappa={kap:.3f}")
        if acc > best_acc:
            best_acc = acc
            os.makedirs("models/ckpts", exist_ok=True)
            torch.save(model.state_dict(), "models/ckpts/sleep_cnn_best.pt")
    print(f"Best acc: {best_acc:.3f}")
