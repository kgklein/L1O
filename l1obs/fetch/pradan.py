"""Local ingestion of already-downloaded ISRO/ISSDC PRADAN Level-2 MAG files.

This provider has no authentication, discovery over HTTP, or downloading support.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .models import FetchResult

PRODUCT = "L2_AL1_MAG"
COMPONENTS = ("Bx_gse", "By_gse", "Bz_gse")
QUALITY = "Quality_flag_10s_data"
USED_VARS = {"time": "time", "bx_gse": "Bx_gse", "by_gse": "By_gse",
             "bz_gse": "Bz_gse", "B_MAG": "sqrt(Bx_gse**2 + By_gse**2 + Bz_gse**2)",
             "quality": QUALITY}


def _utc(value) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if pd.isna(value):
        raise ValueError("Requested times must not be NaT")
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _discover(start: pd.Timestamp, end: pd.Timestamp, directory: Path) -> list[Path]:
    days = pd.date_range(start.normalize(), (end - pd.Timedelta(nanoseconds=1)).normalize(), freq="D")
    files = [directory / f"L2_AL1_MAG_{day.strftime('%Y%m%d')}_V00.nc" for day in days]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"[{PRODUCT}] Missing local daily files in {directory}: {', '.join(missing)}")
    return files


def _values(variable: xr.DataArray, fill: float) -> np.ndarray:
    """Mask the global fill value plus any variable-level validity metadata."""
    values = np.asarray(variable.values, dtype=float).copy()
    invalid = ~np.isfinite(values) | (values == fill)
    for key in ("_FillValue", "missing_value", "FILLVAL"):
        if key in variable.attrs:
            for sentinel in np.asarray(variable.attrs[key]).ravel():
                invalid |= values == float(sentinel)
    for key in ("valid_min", "VALIDMIN"):
        if key in variable.attrs:
            invalid |= values < np.asarray(variable.attrs[key], dtype=float).squeeze()
    for key in ("valid_max", "VALIDMAX"):
        if key in variable.attrs:
            invalid |= values > np.asarray(variable.attrs[key], dtype=float).squeeze()
    if "valid_range" in variable.attrs:
        bounds = np.asarray(variable.attrs["valid_range"], dtype=float)
        if bounds.shape != (2,):
            raise RuntimeError(f"[{PRODUCT}] Invalid valid_range for {variable.name}")
        invalid |= (values < bounds[0]) | (values > bounds[1])
    values[invalid] = np.nan
    return values * variable.attrs.get("scale_factor", 1) + variable.attrs.get("add_offset", 0)


def _read_file(path: Path) -> pd.DataFrame:
    with xr.open_dataset(path, engine="netcdf4", decode_times=False, mask_and_scale=False) as dataset:
        for name in ("time", *COMPONENTS, QUALITY):
            if name not in dataset:
                raise RuntimeError(f"[{PRODUCT}] Missing variable {name} in {path.name}")
        time = dataset["time"]
        if time.ndim != 1 or time.size == 0:
            raise RuntimeError(f"[{PRODUCT}] time must be a nonempty (N,) array")
        # The actual file has separate Bx/By/Bz dimensions: align by sample order,
        # never by xarray dimension names (which would broadcast these arrays).
        for name in (*COMPONENTS, QUALITY):
            if dataset[name].shape != (time.size,):
                raise RuntimeError(f"[{PRODUCT}] {name} must have shape ({time.size},)")
        if str(time.attrs.get("units", "")).strip().casefold() != "seconds (unix time)":
            raise RuntimeError(f"[{PRODUCT}] Unsupported time units: {time.attrs.get('units')!r}")
        fill = float(dataset.attrs.get("Fill_value", -9999.0))
        if not np.isfinite(fill):
            raise RuntimeError(f"[{PRODUCT}] Invalid global Fill_value")
        seconds = _values(time, fill)
        if not np.isfinite(seconds).all():
            raise RuntimeError(f"[{PRODUCT}] Invalid Unix timestamps in {path.name}")
        index = pd.DatetimeIndex(pd.to_datetime(seconds, unit="s", origin="unix", utc=True), name="time")
        if index.hasnans or index.has_duplicates:
            raise RuntimeError(f"[{PRODUCT}] Invalid or duplicate timestamps in {path.name}")
        vector = np.column_stack([_values(dataset[name], fill) for name in COMPONENTS])
        quality = _values(dataset[QUALITY], fill)
        vector[~np.isfinite(quality) | (quality != 1), :] = np.nan
        # This product has no archive magnitude. Derive it only AFTER masking;
        # a missing component leaves the magnitude missing, without filling it.
        magnitude = np.sqrt(np.sum(vector ** 2, axis=1))
        frame = pd.DataFrame({"bx_gse": vector[:, 0], "by_gse": vector[:, 1],
                              "bz_gse": vector[:, 2], "b_mag": magnitude}, index=index)
    frame.attrs = {"coordinate_system": "GSE", "units": "nT",
                   "magnitude_source": "derived from Level-2 GSE vector"}
    return frame


def fetch_pradan_product(product: str, start: pd.Timestamp, end: pd.Timestamp,
                         cache_dir: Path, force: bool = False) -> FetchResult:
    """Read local V00 daily MAG files directly from cache_dir on [start, end).

    Native 10-second samples and invalid rows are retained. Every requested day
    requires its local file. There is no network access or additional interval
    cache; files are reread on every call, so force has no effect for this provider.
    """
    if product != PRODUCT:
        raise ValueError(f"Unsupported local PRADAN product {product!r}")
    start, end = _utc(start), _utc(end)
    if end <= start:
        raise ValueError("Requested end must be after start")
    files = _discover(start, end, Path(cache_dir))
    frame = pd.concat([_read_file(path) for path in files]).sort_index()
    if frame.index.has_duplicates:
        raise RuntimeError(f"[{PRODUCT}] Duplicate timestamps across local daily files")
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    if frame.empty:
        raise RuntimeError(f"[{PRODUCT}] No samples in requested interval")
    return FetchResult(frame, product, USED_VARS.copy())
