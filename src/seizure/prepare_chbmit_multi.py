import yaml, os, glob, json, re
from src.data.chbmit_multi import parse_summaries, edf_to_windows_multi

if __name__ == "__main__":
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    raw = cfg["chb_mit"]["raw_dir"]
    out = cfg["chb_mit"]["processed_dir"]
    os.makedirs(out, exist_ok=True)
    summ = parse_summaries(raw)
    # build index
    idx = {}
    for p in glob.glob(os.path.join(raw, "**", "*.edf"), recursive=True):
        idx[os.path.basename(p)] = p
    mani = []
    for edf, spans in summ.items():
        if edf not in idx:
            print(f"[WARN] EDF missing: {edf}"); continue
        try:
            X, y, chs, fs = edf_to_windows_multi(idx[edf], spans, cfg)
            base = os.path.splitext(edf)[0]
            import numpy as np
            np.savez_compressed(os.path.join(out,f"{base}_multi.npz"), X=X, y=y, chs=chs, fs=fs)
            m = re.match(r"(chb\d+)_", base, re.I)
            pid = m.group(1).lower() if m else "unknown"
            mani.append({"edf": edf, "n": int(len(X)), "pos": int(y.sum()), "chs": chs, "fs": fs, "patient": pid})
            print(f"[OK] {edf}: {len(X)} windows, chs={chs}")
        except Exception as e:
            print(f"[WARN] {edf} failed: {e}")
    with open(os.path.join(out,"manifest_multi.json"),"w") as f:
        json.dump(mani, f, indent=2)
