"""NOAA/NCEI Space Weather Portal daily science-quality magnetic products."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd
import xarray as xr

from .models import FetchResult

API_URL = "https://www.ncei.noaa.gov/cloud-access/space-weather-portal/api/v1/files"
PRODUCT = "sci_mag-l3_solar1"
TIME_UNITS = "Microseconds since 1958-01-01T00:00:00, corrected for leap seconds"
USED_VARS = {"time": "time_sec", "B_GSE": "b_gse_sec",
             "B_MAG": "b_gse_sphr_sec", "quality": "flags_summary"}


def _utc(value) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if pd.isna(value):
        raise ValueError("Requested times must not be NaT")
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _discover(start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    selected, missing = [], []
    for day in pd.date_range(start.normalize(), (end - pd.Timedelta(nanoseconds=1)).normalize(), freq="D"):
        next_day = day + pd.Timedelta(days=1)
        query = urlencode({"prod": PRODUCT, "sat": "SOLAR-1",
                           "start_time": day.isoformat(), "end_time": next_day.isoformat(),
                           "limit": 1000})
        with urlopen(f"{API_URL}?{query}", timeout=60) as response:
            payload = json.load(response)
        if (not isinstance(payload, dict) or not isinstance(payload.get("status"), dict)
                or payload["status"].get("code") != 200 or not isinstance(payload.get("data"), list)):
            raise RuntimeError(f"[{PRODUCT}] Unsuccessful or malformed NOAA /files response")
        if len(payload["data"]) >= 1000:
            raise RuntimeError(f"[{PRODUCT}] NOAA discovery limit reached for {day.date()}")
        candidates = []
        for entry in payload["data"]:
            if not isinstance(entry, dict):
                raise RuntimeError(f"[{PRODUCT}] Malformed NOAA file metadata")
            if entry.get("product") != PRODUCT or entry.get("satellite") != "SOLAR-1":
                continue
            try:
                first, last = _utc(entry["time_coverage_start"]), _utc(entry["time_coverage_end"])
                link = entry["file_link"]
                identifier = entry["id"]
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"[{PRODUCT}] Malformed NOAA file metadata") from error
            if not isinstance(link, str) or not urlparse(link).path.endswith(".nc"):
                continue
            if last <= first:
                raise RuntimeError(f"[{PRODUCT}] Invalid file coverage for {identifier}")
            if first >= min(end, next_day) or last <= max(start, day):
                continue
            # A selected daily product must cover this day's requested portion.
            if first > max(start, day) or last < min(end, next_day):
                continue
            candidates.append(entry)
        if not candidates:
            missing.append(str(day.date()))
            continue
        if len(candidates) > 1:
            revisions = [re.search(r"_p(\d{8}T\d{6}Z)(?:_|\.)", c["id"]) for c in candidates]
            if any(match is None for match in revisions):
                raise RuntimeError(f"[{PRODUCT}] Ambiguous file revisions for {day.date()}")
            stamps = [match.group(1) for match in revisions]
            latest = max(stamps)
            candidates = [c for c, stamp in zip(candidates, stamps) if stamp == latest]
            if len(candidates) != 1:
                raise RuntimeError(f"[{PRODUCT}] Ambiguous file revisions for {day.date()}")
        if candidates[0] not in selected:
            selected.append(candidates[0])
    if missing:
        raise RuntimeError(f"[{PRODUCT}] No daily file for requested days: {', '.join(missing)}")
    return selected


def _download(entry: dict, directory: Path, force: bool) -> Path:
    filename = Path(urlparse(entry["file_link"]).path).name
    target = directory / filename
    if target.exists() and not force:
        return target
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            with urlopen(entry["file_link"], timeout=120) as response:
                shutil.copyfileobj(response, stream)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def _values(variable: xr.DataArray) -> np.ndarray:
    """Apply raw NetCDF validity metadata before scaling or normalization."""
    raw = np.asarray(variable.values)
    values = raw.astype(float).copy()
    attrs = variable.attrs
    invalid = ~np.isfinite(values)
    fills = [attrs[key] for key in ("_FillValue", "missing_value", "FILLVAL") if key in attrs]
    if not fills:
        fills = [-9999.0]
    for fill in fills:
        for sentinel in np.asarray(fill).ravel():
            invalid |= raw == (np.asarray(sentinel, dtype=raw.dtype)
                               if raw.dtype.kind == "f" else sentinel)
    for keys, comparison in [(("valid_min", "VALIDMIN"), np.less),
                             (("valid_max", "VALIDMAX"), np.greater)]:
        for key in keys:
            if key in attrs:
                invalid |= comparison(values, np.asarray(attrs[key]).squeeze())
    if "valid_range" in attrs:
        bounds = np.asarray(attrs["valid_range"])
        if bounds.shape[0] != 2:
            raise ValueError(f"Invalid valid_range for {variable.name}")
        invalid |= (values < bounds[0]) | (values > bounds[1])
    values[invalid] = np.nan
    return values * attrs.get("scale_factor", 1) + attrs.get("add_offset", 0)


def _read_file(path: Path) -> pd.DataFrame:
    with xr.open_dataset(path, engine="netcdf4", decode_times=False, mask_and_scale=False) as dataset:
        for name in USED_VARS.values():
            if name not in dataset:
                raise RuntimeError(f"[{PRODUCT}] Missing variable {name} in {path.name}")
        time = dataset["time_sec"]
        if time.ndim != 1 or time.size == 0:
            raise RuntimeError(f"[{PRODUCT}] time_sec must be a nonempty (N,) array")
        for name in ("b_gse_sec", "b_gse_sphr_sec", "flags_summary"):
            variable = dataset[name]
            shape = (time.size,) if name == "flags_summary" else (time.size, 3)
            if variable.shape != shape or variable.dims[0] != time.dims[0]:
                raise RuntimeError(f"[{PRODUCT}] {name} must have aligned shape {shape}")
        units = time.attrs.get("units")
        if units != TIME_UNITS:
            raise RuntimeError(f"[{PRODUCT}] Unsupported time_sec units: {units!r}")
        counts = _values(time)
        if not np.isfinite(counts).all():
            raise RuntimeError(f"[{PRODUCT}] Invalid time_sec timestamps")
        clean_time = xr.DataArray(counts, dims=time.dims,
                                 attrs={"units": "microseconds since 1958-01-01T00:00:00"})
        decoded = xr.decode_cf(xr.Dataset({"time_sec": clean_time}))["time_sec"].values
        index = pd.DatetimeIndex(pd.to_datetime(decoded, utc=True), name="time")
        if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
            raise RuntimeError(f"[{PRODUCT}] Invalid or duplicate time_sec timestamps")
        vector = _values(dataset["b_gse_sec"])
        magnitude = _values(dataset["b_gse_sphr_sec"])[:, 0]
        quality = _values(dataset["flags_summary"])
        frame = pd.DataFrame({"bx_gse": vector[:, 0], "by_gse": vector[:, 1],
                              "bz_gse": vector[:, 2], "b_mag": magnitude}, index=index)
        frame.loc[~np.isfinite(quality) | (quality != 0), :] = np.nan
    frame.attrs = {"coordinate_system": "GSE", "units": "nT"}
    return frame


def fetch_ncei_product(product: str, start: pd.Timestamp, end: pd.Timestamp,
                       cache_dir: Path, force: bool = False) -> FetchResult:
    """Retrieve native SOLAR-1 GSE data on [start, end), requiring every daily file.

    Archive UTC microsecond counts need no additional leap-second correction.
    Bad quality rows are retained as NaNs; no interpolation or resampling occurs.
    """
    if product != PRODUCT:
        raise ValueError(f"Unsupported NOAA/NCEI product {product!r}")
    start, end = _utc(start), _utc(end)
    if end <= start:
        raise ValueError("Requested end must be after start")
    directory = Path(cache_dir) / "ncei" / product
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / f"{product}_mag_v1_{start.value}_{end.value}.parquet"
    if cache.exists() and not force:
        return FetchResult(pd.read_parquet(cache), product, {"_cached": "true"})
    entries = _discover(start, end)
    frame = pd.concat([_read_file(_download(entry, directory, force)) for entry in entries]).sort_index()
    duplicate_times = frame.index[frame.index.duplicated()].unique()
    for timestamp in duplicate_times:
        rows = frame.loc[[timestamp]].to_numpy()
        if not np.all((rows == rows[0]) | (np.isnan(rows) & np.isnan(rows[0]))):
            raise RuntimeError(f"[{PRODUCT}] Conflicting duplicate records at {timestamp}")
    frame = frame.loc[~frame.index.duplicated()]
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    if frame.empty:
        raise RuntimeError(f"[{PRODUCT}] No samples in requested interval")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
        frame.to_parquet(temporary)
        os.replace(temporary, cache)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return FetchResult(frame, product, USED_VARS.copy())
