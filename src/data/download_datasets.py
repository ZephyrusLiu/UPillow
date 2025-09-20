import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import yaml
import wfdb

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _filter_records(records: Iterable[str], prefix: str) -> List[str]:
    pref = prefix.rstrip("/") + "/"
    return [r for r in records if r.startswith(pref)]


def _download_records(db: str, destination: Path, records: List[str]) -> None:
    if not records:
        raise RuntimeError(f"No records matched for '{db}' with the provided filters.")

    _ensure_dir(destination)
    print(f"Downloading {len(records)} records from '{db}' into '{destination}'.")
    wfdb.dl_database(db, dl_dir=str(destination), records=records, keep_subdirs=True)


def download_sleep_edf(subset: str, destination: Path) -> None:
    try:
        records = wfdb.get_record_list("sleep-edfx")
    except Exception as exc:  # pragma: no cover - network failure path
        raise RuntimeError(
            "Unable to query record list from PhysioNet for sleep-edfx. "
            "Please ensure you have network access and, if required, have "
            "authenticated with PhysioNet."
        ) from exc

    prefix = "sleep-cassette" if subset == "cassette" else "sleep-telemetry"
    subset_records = _filter_records(records, prefix)
    if not subset_records:
        raise RuntimeError(
            f"PhysioNet record list for sleep-edfx did not contain entries under '{prefix}'."
        )

    _download_records("sleep-edfx", destination, subset_records)


def download_chb_mit(destination: Path) -> None:
    try:
        records = wfdb.get_record_list("chbmit")
    except Exception as exc:  # pragma: no cover - network failure path
        raise RuntimeError(
            "Unable to query record list from PhysioNet for chbmit. "
            "Please ensure you have network access and access permissions."
        ) from exc

    _download_records("chbmit", destination, records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download required PhysioNet datasets.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file containing raw_dir locations.",
    )
    parser.add_argument(
        "--sleep-cassette",
        action="store_true",
        help="Download the Sleep-EDF Sleep Cassette subset.",
    )
    parser.add_argument(
        "--sleep-telemetry",
        action="store_true",
        help="Download the Sleep-EDF Sleep Telemetry subset.",
    )
    parser.add_argument(
        "--chb-mit",
        action="store_true",
        help="Download the CHB-MIT Scalp EEG dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = {
        "sleep_cassette": args.sleep_cassette,
        "sleep_telemetry": args.sleep_telemetry,
        "chb_mit": args.chb_mit,
    }
    if not any(targets.values()):
        targets = {k: True for k in targets}

    cfg = _load_config(args.config)
    try:
        if targets["sleep_cassette"]:
            dest = Path(cfg["sleep_edf"]["raw_dir"])
            print(f"Preparing to download Sleep Cassette into '{dest}'.")
            download_sleep_edf("cassette", dest)
        if targets["sleep_telemetry"]:
            dest = Path(cfg["sleep_edf"]["raw_dir"])
            print(f"Preparing to download Sleep Telemetry into '{dest}'.")
            download_sleep_edf("telemetry", dest)
        if targets["chb_mit"]:
            dest = Path(cfg["chb_mit"]["raw_dir"])
            print(f"Preparing to download CHB-MIT into '{dest}'.")
            download_chb_mit(dest)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())