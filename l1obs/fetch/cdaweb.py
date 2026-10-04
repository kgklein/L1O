from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from cdasws import CdasWs
from cdasws.datarepresentation import DataRepresentation

from .variable_map import VAR_CANDIDATES, DATASET_LOGICAL_VARS, MAGNETIC_PRODUCTS
from .models import FetchResult

import logging
log = logging.getLogger("l1obs.fetch.cdaweb")


def _pick_var(available: List[str], candidates: List[str]) -> Optional[str]:
    aset = set(available)
    for c in candidates:
        if c in aset:
            return c
    return None


def _resolve_vars(cdas: CdasWs, dataset_id: str) -> Dict[str, str]:
    if dataset_id in MAGNETIC_PRODUCTS:
        return MAGNETIC_PRODUCTS[dataset_id].variables()
    # returns mapping logical_name -> actual variable name in dataset
    avail = cdas.get_variables(dataset_id)
    # cdasws returns list of dicts sometimes; normalize to names
    if len(avail) > 0 and isinstance(avail[0], dict) and "Name" in avail[0]:
        available_names = [v["Name"] for v in avail]
    else:
        available_names = list(avail)

    logical = DATASET_LOGICAL_VARS.get(dataset_id, [])
    used: Dict[str, str] = {}
    for lname in logical:
        vname = _pick_var(available_names, VAR_CANDIDATES.get(lname, [lname]))
        if vname is not None:
            used[lname] = vname

    if not used:
        raise RuntimeError(
            f"[{dataset_id}] Could not match any variables.\n"
            f"Available variables include:\n  " + "\n  ".join(available_names[:200])
        )
    return used


def _cache_path(cache_dir: Path, dataset_id: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    stamp = f"{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}"
    version = "_mag_v1" if dataset_id in MAGNETIC_PRODUCTS else ""
    return cache_dir / f"{dataset_id}{version}_{stamp}.parquet"


def _mask_invalid(variable, dataset_id: str) -> np.ndarray:
    """Apply CDF validity attributes before discarding the variable metadata."""
    raw = np.asarray(variable.values)
    values = raw.astype(float).copy()
    invalid = ~np.isfinite(values)
    attrs = variable.attrs
    fill = attrs.get("FILLVAL")
    if fill is not None:
        # Compare at the source precision, notably for CDF_REAL4 sentinels.
        invalid |= raw == np.asarray(fill, dtype=raw.dtype).squeeze()
    else:
        invalid |= (values == np.float64(-1e31)) | (values == float(np.float32(-1e31)))
    for attribute, comparison in [("VALIDMIN", np.less), ("VALIDMAX", np.greater)]:
        if attribute in attrs:
            invalid |= comparison(values, np.asarray(attrs[attribute], dtype=float).squeeze())
        else:
            log.warning("[%s] %s lacks %s; no corresponding bound applied",
                        dataset_id, variable.name, attribute)
    values[invalid] = np.nan
    return values


def _fetch_magnetic(
    cdas: CdasWs, dataset_id: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    product = MAGNETIC_PRODUCTS[dataset_id]
    # Dependency epochs are returned with the requested science variables.
    requested = [product.vector, product.magnitude]
    if product.quality is not None:
        requested.append(product.quality)
    status, data = cdas.get_data(
        dataset_id, requested, start.to_pydatetime(), end.to_pydatetime(),
        dataRepresentation=DataRepresentation.XARRAY,
    )
    http_status = status.get("http", {}).get("status_code")
    if http_status != 200 or data is None:
        raise RuntimeError(f"[{dataset_id}] No magnetic data for {start} to {end} "
                           f"(HTTP {http_status}; {status.get('cdas', {})}).")
    required = [product.time, *requested]
    missing = [name for name in required if name not in data]
    if missing:
        raise RuntimeError(f"[{dataset_id}] Missing required variables: {', '.join(missing)}")
    try:
        time = data[product.time]
        index = pd.DatetimeIndex(pd.to_datetime(time.values, utc=True), name="time")
        if time.ndim != 1 or index.hasnans or len(index) == 0:
            raise ValueError("time must be a nonempty one-dimensional array without NaT")
        arrays = {}
        for name in requested:
            variable = data[name]
            expected = (len(index), 3) if name == product.vector else (len(index),)
            if variable.shape != expected or variable.dims[0] != time.dims[0]:
                raise ValueError(f"{name} must align with {product.time} and have shape {expected}")
            arrays[name] = _mask_invalid(variable, dataset_id)
        vector = arrays[product.vector]
        df = pd.DataFrame({"bx_gse": vector[:, 0], "by_gse": vector[:, 1],
                           "bz_gse": vector[:, 2], "b_mag": arrays[product.magnitude]},
                          index=index)
        if product.quality is not None:
            df.loc[arrays[product.quality] != 0, :] = np.nan
    except (TypeError, ValueError, IndexError) as exc:
        raise RuntimeError(f"[{dataset_id}] Malformed magnetic product: {exc}") from exc
    return df


def fetch_cdaweb_dataset(
    dataset_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
    force: bool = False,
) -> FetchResult:
    """Fetch native samples; magnetic products use a UTC time index and
    bx_gse, by_gse, bz_gse, b_mag columns in nT, with invalid values as NaN.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = _cache_path(cache_dir, dataset_id, start, end)

    if cpath.exists() and not force:
        log.info("[%s] cache hit: %s", dataset_id, cpath.name)
        df = pd.read_parquet(cpath)
        # parquet won’t preserve index name reliably across all environments
        df.index = pd.to_datetime(df.index, utc=True)
        if dataset_id in MAGNETIC_PRODUCTS:
            df.index.name = "time"
        return FetchResult(df=df, dataset_id=dataset_id, used_vars={"_cached": "true"})

    cdas = CdasWs()
    used = _resolve_vars(cdas, dataset_id)
    if dataset_id in MAGNETIC_PRODUCTS:
        df = _fetch_magnetic(cdas, dataset_id, start, end)
        df.to_parquet(cpath)
        return FetchResult(df=df, dataset_id=dataset_id, used_vars=used)
    actual_vars = list(used.values())

    # Ask for pandas output
    # cdasws is happiest with naive Python datetimes in UTC
    t1 = start.to_pydatetime().replace(tzinfo=None)
    t2 = end.to_pydatetime().replace(tzinfo=None)

    try:
        # Newer/alternate signature: (dataset, variables, time1, time2, ...)
        data = cdas.get_data(dataset_id, actual_vars, t1, t2, data_type="pandas")
    except TypeError:
        # Other signature: (dataset, time1, time2, variables, ...)
        data = cdas.get_data(dataset_id, t1, t2, actual_vars, data_type="pandas")

    if data is None or "data" not in data or data["data"] is None:
        raise RuntimeError(f"[{dataset_id}] No data returned for {start} to {end}.")

    df: pd.DataFrame = data["data"].copy()
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")

    # Debug: print columns once per dataset
    print(f"[{dataset_id}] columns: {list(df.columns)[:30]}{' ...' if len(df.columns)>30 else ''}")


    # Save cache
    df.to_parquet(cpath)

    return FetchResult(df=df, dataset_id=dataset_id, used_vars=used)
