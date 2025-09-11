import yaml
from src.data.chbmit import prepare_chbmit

if __name__ == "__main__":
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    prepare_chbmit(cfg)
