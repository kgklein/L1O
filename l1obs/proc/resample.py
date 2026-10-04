from __future__ import annotations

import pandas as pd


def resample_to_1s(
    df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp,
    preserve_nan_gaps: bool = False,
) -> pd.DataFrame:
    """
    Reindex to 1-second cadence and interpolate in time.
    Assumes df has a UTC DateTimeIndex.
    With preserve_nan_gaps, interpolate only within contiguous valid runs,
    independently for each numeric column, without endpoint extrapolation.
    """
    idx = pd.date_range(t0, t1, freq="1s", tz="UTC", inclusive="left")
    if preserve_nan_gaps:
        source = df.sort_index()
        out = pd.DataFrame(float("nan"), index=idx, columns=source.columns)
        for column in source.columns:
            values = source[column]
            valid = values.notna()
            groups = (~valid).cumsum()
            for _, run in values[valid].groupby(groups[valid]):
                target = idx[(idx >= run.index[0]) & (idx <= run.index[-1])]
                interpolated = run.reindex(run.index.union(target)).sort_index()
                out.loc[target, column] = interpolated.interpolate(method="time").reindex(target)
        out.index.name = df.index.name
        return out
    # align, then interpolate only numeric columns
    out = df.reindex(df.index.union(idx)).sort_index()
    out = out.interpolate(method="time", limit_direction="both")
    out = out.reindex(idx)
    return out
