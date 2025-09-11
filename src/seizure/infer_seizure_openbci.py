import argparse, torch, numpy as np
from src.models.cnn_1d_multi import TinySleepCNNMulti
from src.data.openbci16 import prepare_openbci_npz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="OpenBCI .txt or BrainFlow .csv")
    ap.add_argument("--ckpt", default="models/ckpts/chbmit_cnn_multi_best.pt")
    ap.add_argument("--out", default="outputs/seizure_probs.npy")
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--vote_k", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    info = prepare_openbci_npz(args.input, "tmp_openbci_seiz.npz",
                               task="seizure", seiz_fs=args.fs, seiz_win=args.win, seiz_hop=args.hop)
    npz = np.load("tmp_openbci_seiz.npz", allow_pickle=True)
    X = npz["X"]  # (N,C,T)
    in_ch = X.shape[1]; in_len = X.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinySleepCNNMulti(n_classes=2, in_len=in_len, in_ch=in_ch).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd); model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, X.shape[0]):
            xb = torch.from_numpy(X[i:i+1]).to(device)
            p = torch.softmax(model(xb), dim=1)[0,1].cpu().numpy().item()
            probs.append(p)
    probs = np.array(probs, dtype=np.float32)
    # majority voting over last k after threshold binarization
    import collections
    preds = (probs >= args.thr).astype(np.int32)
    mv = []
    dq = collections.deque(maxlen=args.vote_k)
    for p in preds:
        dq.append(p)
        vals, cnts = np.unique(np.array(dq), return_counts=True)
        mv.append(int(vals[np.argmax(cnts)]))
    import os; os.makedirs("outputs", exist_ok=True)
    np.save(args.out, probs)
    print(f"Done. Windows: {len(probs)}, mean prob={probs.mean():.3f}, mv_pos_rate={np.mean(mv):.3f}")

if __name__ == "__main__":
    main()
