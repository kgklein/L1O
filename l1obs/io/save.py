from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np



def save_hdf5(df: pd.DataFrame, outpath: Path, key: str = "l1obs") -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    # Use table format so we can append later if desired
    df.to_hdf(outpath, key=key, mode="w", format="table", complevel=5, complib="zlib")

def save_cdf(df: pd.DataFrame, outpath: Path) -> None:
    """
    Save merged dataframe to a NASA CDF file.

    - Time index becomes variable "Epoch" (TT2000)
    - Each numeric column becomes its own variable
    - NaNs become a fill value (default: -1e31 for float, min int for ints)
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Ensure UTC DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    # Convert to TT2000
    # cdflib expects list-like of datetime components; easiest: use pandas -> python datetime
    dt_list = [t.to_pydatetime().replace(tzinfo=None) for t in idx.to_pydatetime()]
    epoch_tt2000 = cdfepoch.compute_tt2000(dt_list)

    # Create CDF (overwrite)
    if outpath.exists():
        outpath.unlink()

    cdf = cdfwrite.CDF(str(outpath))

    # --- Write Epoch variable ---
    cdf.write_var(
        var_name="Epoch",
        var_data=epoch_tt2000,
        var_type=33,  # CDF_TIME_TT2000
        var_spec={
            "recVary": True,
            "dimSizes": [],
        },
        var_attrs={
            "CATDESC": "Time (TT2000)",
            "FIELDNAM": "Epoch",
            "VAR_TYPE": "support_data",
        },
    )

    # --- Write data variables ---
    # Keep only numeric columns; drop object columns (vectors-in-objects, etc.)
    num_df = df.copy()

    # Convert bool -> int8 (CDF doesn’t love bool)
    for col in num_df.columns:
        if pd.api.types.is_bool_dtype(num_df[col]):
            num_df[col] = num_df[col].astype(np.int8)

    numeric_cols = [c for c in num_df.columns if pd.api.types.is_numeric_dtype(num_df[c])]
    if not numeric_cols:
        raise RuntimeError("No numeric columns found to write to CDF.")

    for col in numeric_cols:
        arr = np.asarray(num_df[col].to_numpy())

        # Choose fill value based on dtype
        if np.issubdtype(arr.dtype, np.floating):
            fill = np.float64(-1e31)
            arr = arr.astype(np.float64)
            arr = np.where(np.isfinite(arr), arr, fill)
            cdf_type = 45  # CDF_DOUBLE
        elif np.issubdtype(arr.dtype, np.integer):
            fill = np.iinfo(arr.dtype).min
            arr = np.where(pd.isna(arr), fill, arr)
            cdf_type = 4  # CDF_INT4 (we’ll coerce to int32)
            arr = arr.astype(np.int32)
        else:
            # Skip anything weird
            continue

        cdf.write_var(
            var_name=col,
            var_data=arr,
            var_type=cdf_type,
            var_spec={
                "recVary": True,
                "dimSizes": [],
            },
            var_attrs={
                "FIELDNAM": col,
                "CATDESC": col,
                "DEPEND_0": "Epoch",
                "FILLVAL": fill,
                "VAR_TYPE": "data",
            },
        )

    cdf.close()
