import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from src.models.cnn_1d_multi import TinySleepCNNMulti
from src.sleep.data_utils import load_sleepedf_multi_npz


class NPZMultiDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype("float32")
        self.y = y.astype("int64")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["sleep_edf"]["processed_dir"]
    data = load_sleepedf_multi_npz(processed_dir)
    if not data:
        raise SystemExit("No processed Sleep-EDF files found. Run the preprocessing scripts first.")

    X = np.concatenate([item[1] for item in data], axis=0)
    y = np.concatenate([item[2] for item in data], axis=0)

    in_ch = X.shape[1]
    in_len = X.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinySleepCNNMulti(n_classes=5, in_len=in_len, in_ch=in_ch).to(device)

    ckpt_path = "models/ckpts/sleep_cnn_multi_best.pt"
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"Checkpoint '{ckpt_path}' not found. Train the multi-channel CNN with scripts/07_train_sleep_cnn_multi.sh first."
        )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dl = DataLoader(NPZMultiDataset(X, y), batch_size=128, shuffle=False)
    yp_all, yt_all = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = model(xb)
            yp_all.append(logits.argmax(1).cpu().numpy())
            yt_all.append(yb.numpy())

    yp_all = np.concatenate(yp_all)
    yt_all = np.concatenate(yt_all)

    print("Accuracy:", accuracy_score(yt_all, yp_all))
    print("Macro F1:", f1_score(yt_all, yp_all, average="macro"))
    print("Cohen Kappa:", cohen_kappa_score(yt_all, yp_all))
    print("Confusion Matrix:\n", confusion_matrix(yt_all, yp_all))
    print(classification_report(yt_all, yp_all, digits=3))