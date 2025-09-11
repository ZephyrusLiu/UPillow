import os, glob, yaml, json
from src.data.sleepedf_multi import extract_epochs_multi

if __name__ == "__main__":
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    raw_dir = cfg["sleep_edf"]["raw_dir"]
    out_dir = cfg["sleep_edf"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)
    import glob
    edfs = glob.glob(os.path.join(raw_dir, "**", "*.edf"), recursive=True)
    manifest = []
    for eeg_path in edfs:
        hyp_path = ""  # rely on annotations inside if not present separately
        try:
            X, y, classes, chs, fs = extract_epochs_multi(eeg_path, hyp_path, cfg, os.path.basename(eeg_path))
            if len(X)==0: 
                continue
            base = os.path.splitext(os.path.basename(eeg_path))[0]
            import numpy as np
            np.savez_compressed(os.path.join(out_dir,f"{base}_multi.npz"), X=X, y=y, classes=classes, chs=chs, fs=fs)
            manifest.append({"file": base, "n": int(len(X)), "chs": chs, "fs": fs})
            print(f"[OK] {base}: {len(X)} epochs, chs={chs}")
        except Exception as e:
            print(f"[WARN] {eeg_path} failed: {e}")
    with open(os.path.join(out_dir,"manifest_multi.json"),"w") as f:
        json.dump(manifest, f, indent=2)
