import os, json, yaml
from src.data.sleepedf import save_processed

if __name__ == "__main__":
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    raw_dir = cfg["sleep_edf"]["raw_dir"]
    out_dir = cfg["sleep_edf"]["processed_dir"]
    save_processed(raw_dir, out_dir, cfg)
