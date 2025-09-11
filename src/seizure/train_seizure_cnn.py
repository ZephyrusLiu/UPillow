import os, yaml, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.models.cnn_1d import TinySleepCNN
from tqdm import tqdm

class NPZBin(Dataset):
    def __init__(self, X, y):
        self.X = X[:,None,:].astype("float32"); self.y = y.astype("int64")
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        import torch
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

if __name__ == "__main__":
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    pdir = cfg["sleep_edf"]["processed_dir"]
    npz_path = os.path.join(pdir, "synthetic_seizure", "synthetic_seizure.npz")
    if not os.path.exists(npz_path):
        raise SystemExit("Run simulate.py first.")
    data = np.load(npz_path, allow_pickle=True)
    X, y, fs = data["X"], data["y"], int(data["fs"])

    # split
    n = len(y); n_te = max(100, n//5)
    Xtr, ytr = X[:-n_te], y[:-n_te]
    Xte, yte = X[-n_te:], y[-n_te:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinySleepCNN(n_classes=2, in_len=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()

    dl_tr = DataLoader(NPZBin(Xtr,ytr), batch_size=64, shuffle=True)
    dl_te = DataLoader(NPZBin(Xte,yte), batch_size=128, shuffle=False)

    for ep in range(10):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        yp_all, yt_all = [], []
        with torch.no_grad():
            for xb, yb in dl_te:
                xb = xb.to(device)
                yp = model(xb).argmax(1).cpu().numpy()
                yp_all.append(yp); yt_all.append(yb.numpy())
        import numpy as np
        yp_all = np.concatenate(yp_all); yt_all = np.concatenate(yt_all)
        acc = accuracy_score(yt_all, yp_all); f1 = f1_score(yt_all, yp_all)
        print(f"Epoch {ep+1}: acc={acc:.3f}, f1={f1:.3f}")
