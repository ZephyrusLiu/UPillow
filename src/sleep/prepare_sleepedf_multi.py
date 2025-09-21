import json
import os
from pathlib import Path

import yaml

from src.data.sleepedf import load_sleepedf_subjects
from src.data.sleepedf_multi import extract_epochs_multi


if __name__ == "__main__":
    # 读取配置文件
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["sleep_edf"]["raw_dir"]
    out_dir = cfg["sleep_edf"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 自动配对 EEG 和 Hypnogram
    pairs = load_sleepedf_subjects(raw_dir)

    manifest = []
    for sid, paths in sorted(pairs.items()):
        eeg_path = paths["eeg"]
        hyp_path = paths.get("hyp", "") or ""

        try:
            X, y, classes, chs, fs = extract_epochs_multi(
                eeg_path,
                hyp_path,
                cfg,
                os.path.basename(eeg_path),
            )
            if len(X) == 0:
                continue

            base = Path(eeg_path).stem

            import numpy as np
            np.savez_compressed(
                os.path.join(out_dir, f"{base}_multi.npz"),
                X=X,
                y=y,
                classes=classes,
                chs=chs,
                fs=fs,
            )

            manifest.append({"file": base, "n": int(len(X)), "chs": chs, "fs": fs})
            print(f"[OK] {base}: {len(X)} epochs, chs={chs}")

        except Exception as e:
            print(f"[WARN] {eeg_path} failed: {e}")

    # 保存清单
    with open(os.path.join(out_dir, "manifest_multi.json"), "w") as f:
        json.dump(manifest, f, indent=2)
