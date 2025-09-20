import argparse
import sys
from pathlib import Path
from typing import Iterable

import yaml
import wfdb
from mne.datasets.sleep_physionet import age, temazepam


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_wfdb_files(db: str, destination: Path, records: Iterable[str]) -> None:
    records = list(records)
    if not records:
        raise RuntimeError(f"No records matched for '{db}' with the provided filters.")

    _ensure_dir(destination)
    print(f"Downloading {len(records)} records from '{db}' into '{destination}'.")
    wfdb.dl_files(db, dl_dir=str(destination), files=records, keep_subdirs=True)


def download_sleep_edf(subset: str, destination: Path) -> None:
    _ensure_dir(destination)
    try:
        if subset == "cassette":
            # All publicly released subjects excluding the documented gaps.
            subjects = [
                s
                for s in range(83)
                if s not in {39, 68, 69, 78, 79}
            ]
            pairs = age.fetch_data(
                subjects=subjects,
                recording=(1, 2),
                path=str(destination),
                force_update=False,
                on_missing="ignore",
                verbose="ERROR",
            )
        else:
            # Placebo nights from the temazepam (Sleep Telemetry) study.
            pairs = temazepam.fetch_data(
                subjects=list(range(22)),
                path=str(destination),
                force_update=False,
                verbose="ERROR",
            )
    except Exception as exc:  # pragma: no cover - network failure path
        raise RuntimeError(
            "MNE could not fetch the requested Sleep-EDF subset. "
            "Ensure you accepted the dataset license and have network access."
        ) from exc

    total = sum(len(p) for p in pairs)
    print(
        f"Downloaded {len(pairs)} recordings ({total} files) for Sleep-EDF {subset}."
    )


def download_chb_mit(destination: Path) -> None:
    try:
        records = wfdb.get_record_list("chbmit")
    except Exception as exc:  # pragma: no cover - network failure path
        raise RuntimeError(
            "Unable to query record list from PhysioNet for chbmit. "
            "Please ensure you have network access and access permissions."
        ) from exc

    _download_wfdb_files("chbmit", destination, records)


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