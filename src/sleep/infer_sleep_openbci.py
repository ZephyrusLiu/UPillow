import argparse, torch, numpy as np
from src.models.cnn_1d_multi import TinySleepCNNMulti
from src.data.openbci16 import prepare_openbci_npz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="OpenBCI .txt or BrainFlow .csv")
    ap.add_argument("--ckpt", default="models/ckpts/sleep_cnn_multi_best.pt")
    ap.add_argument("--out", default="outputs/sleep_preds.npy")
    ap.add_argument("--fs", type=float, default=100.0)
    ap.add_argument("--epoch", type=float, default=30.0)
    ap.add_argument("--labels", default="W,N1,N2,N3,REM")
    args = ap.parse_args()

    info = prepare_openbci_npz(args.input, "tmp_openbci_sleep.npz",
                               task="sleep", sleep_fs=args.fs, sleep_win=args.epoch)
    npz = np.load("tmp_openbci_sleep.npz", allow_pickle=True)
    X = npz["X"]  # (N,C,T)
    in_ch = X.shape[1]; in_len = X.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinySleepCNNMulti(n_classes=5, in_len=in_len, in_ch=in_ch).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd); model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0]):
            xb = torch.from_numpy(X[i:i+1]).to(device)
            y = model(xb).argmax(1).cpu().numpy()[0]
            preds.append(int(y))
    preds = np.array(preds, dtype=np.int32)
    import os; os.makedirs("outputs", exist_ok=True)
    np.save(args.out, preds)
    names = args.labels.split(",")
    counts = np.bincount(preds, minlength=5)
    print("Done. Windows:", len(preds), "Counts:", {names[i]: int(counts[i]) for i in range(5)})

if __name__ == "__main__":
    main()
