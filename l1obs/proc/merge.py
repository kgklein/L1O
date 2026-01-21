from __future__ import annotations

from typing import Dict
import pandas as pd


def merge_frames(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Column-wise join on common time index.
    """
    merged = None
    for sc, df in frames.items():
        if merged is None:
            merged = df.copy()
        else:
            merged = merged.join(df, how="outer")
    if merged is None:
        raise RuntimeError("No frames to merge.")
    merged = merged.sort_index()
    return merged
