from __future__ import annotations

import pandas as pd


def resample_to_1s(df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """
    Reindex to 1-second cadence and interpolate in time.
    Assumes df has a UTC DateTimeIndex.
    """
    idx = pd.date_range(t0, t1, freq="1s", tz="UTC", inclusive="left")
    # align, then interpolate only numeric columns
    out = df.reindex(df.index.union(idx)).sort_index()
    out = out.interpolate(method="time", limit_direction="both")
    out = out.reindex(idx)
    return out
