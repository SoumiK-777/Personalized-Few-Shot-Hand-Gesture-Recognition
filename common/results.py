"""Read canonical experiment summaries without duplicate rerun rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULT_KEY = ["method", "backbone", "n_way", "k_shot", "split"]


def load_results(results_dir: Path) -> pd.DataFrame:
    path = Path(results_dir) / "all_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    # Reruns append duplicate configurations. A configuration must appear once
    # in every paper figure/table, using its latest completed row.
    return (
        frame.sort_values("timestamp").drop_duplicates(RESULT_KEY, keep="last").copy()
    )
