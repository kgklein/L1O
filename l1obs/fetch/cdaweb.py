from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from cdasws import CdasWs

from .variable_map import VAR_CANDIDATES, DATASET_LOGICAL_VARS

import logging
log = logging.getLogger("l1obs.fetch.cdaweb")


@dataclass
class FetchResult:
    df: pd.DataFrame
    dataset_id: str
    used_vars: Dict[str, str]  # logical -> actual var name


def _pick_var(available: List[str], candidates: List[str]) -> Optional[str]:
    aset = set(available)
    for c in candidates:
        if c in aset:
            return c
    return None


def _resolve_vars(cdas: CdasWs, dataset_id: str) -> Dict[str, str]:
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
    return cache_dir / f"{dataset_id}_{stamp}.parquet"


def fetch_cdaweb_dataset(
    dataset_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
    force: bool = False,
) -> FetchResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = _cache_path(cache_dir, dataset_id, start, end)

    if cpath.exists() and not force:
        log.info("[%s] cache hit: %s", dataset_id, cpath.name)
        df = pd.read_parquet(cpath)
        # parquet won’t preserve index name reliably across all environments
        df.index = pd.to_datetime(df.index, utc=True)
        return FetchResult(df=df, dataset_id=dataset_id, used_vars={"_cached": "true"})

    cdas = CdasWs()
    used = _resolve_vars(cdas, dataset_id)
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
