import os, glob, yaml, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, f1_score, accuracy_score
from src.models.cnn_1d import TinySleepCNN

class NPZDataset(Dataset):
    def __init__(self, X, y):
        self.X = X[:,None,:].astype("float32")
        self.y = y.astype("int64")
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        import torch
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

if __name__ == "__main__":
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    pdir = cfg["sleep_edf"]["processed_dir"]
    files = sorted(glob.glob(os.path.join(pdir, "*.npz")))
    X = []; y = []
    for f in files:
        if os.path.basename(f).startswith("manifest"):
            continue
        npz = np.load(f, allow_pickle=True)
        X.append(npz["X"]); y.append(npz["y"])
    X = np.concatenate(X, axis=0); y = np.concatenate(y, axis=0)

    in_len = X.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinySleepCNN(n_classes=5, in_len=in_len).to(device)
    model.load_state_dict(torch.load("models/ckpts/sleep_cnn_best.pt", map_location=device))
    model.eval()

    dl = DataLoader(NPZDataset(X,y), batch_size=128, shuffle=False)
    yp_all, yt_all = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = model(xb)
            yp = logits.argmax(1).cpu().numpy()
            yp_all.append(yp); yt_all.append(yb.numpy())
    import numpy as np
    yp_all = np.concatenate(yp_all); yt_all = np.concatenate(yt_all)
    print("Accuracy:", accuracy_score(yt_all, yp_all))
    print("Macro F1:", f1_score(yt_all, yp_all, average="macro"))
    print("Cohen Kappa:", cohen_kappa_score(yt_all, yp_all))
    print("Confusion Matrix:\n", confusion_matrix(yt_all, yp_all))
    print(classification_report(yt_all, yp_all, digits=3))
