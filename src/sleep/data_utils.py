import glob
import os
from typing import List, Tuple, Optional

import numpy as np


def load_sleepedf_multi_npz(processed_dir: str) -> List[Tuple[str, np.ndarray, np.ndarray, Optional[int]]]:
    """Load Sleep-EDF epochs ensuring a channel axis is present.

    Returns a list of tuples ``(subject_id, X, y, fs)`` where ``X`` has
    shape ``(n_epochs, n_channels, n_samples)``.
    """
    files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
    data = []
    for f in files:
        if os.path.basename(f).startswith("manifest"):
            continue
        npz = np.load(f, allow_pickle=True)
        X = npz["X"]
        y = npz["y"]
        if X.ndim == 2:  # single-channel fallback
            X = X[:, None, :]
        fs = int(npz["fs"]) if "fs" in npz else None
        sid = os.path.splitext(os.path.basename(f))[0]
        data.append((sid, X, y, fs))
    return data