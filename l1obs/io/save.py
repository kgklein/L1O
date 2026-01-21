from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_hdf5(df: pd.DataFrame, outpath: Path, key: str = "l1obs") -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    # Use table format so we can append later if desired
    df.to_hdf(outpath, key=key, mode="w", format="table", complevel=5, complib="zlib")
