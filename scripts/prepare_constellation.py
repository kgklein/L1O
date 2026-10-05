#!/usr/bin/env python3
"""Prepare one UTC day's local inputs for ``l1obs constellation``.

Run with L1O installed (``python -m pip install -e .``). Remote magnetic products
use the existing provider caches; processed magnetic/position tables are saved
separately, without resampling. Aditya-L1 MAG and positions come from its local
PRADAN Level-2 daily file. Other ephemerides are requested from SSCWeb in GSE,
with five minutes of padding to bracket the day's boundaries when available.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import logging
import os
from pathlib import Path
import shlex
import sys
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

from l1obs.config import SPACECRAFT
from l1obs.fetch.magnetic import fetch_magnetic_field
from l1obs.fetch.sscweb import fetch_ephemeris

log = logging.getLogger("prepare_constellation")
MAGNETIC = ("bx_gse", "by_gse", "bz_gse", "b_mag")
POSITIONS = ("x_km", "y_km", "z_km")


def _aditya_positions(path: Path) -> pd.DataFrame:
    """Read independent position variables; MAG quality flags do not flag orbit data."""
    with xr.open_dataset(path, engine="netcdf4", decode_times=False, mask_and_scale=False) as ds:
        for name in ("time", "x_gse", "y_gse", "z_gse"):
            if name not in ds:
                raise ValueError(f"{path.name}: missing {name}")
        if ds.time.ndim != 1 or str(ds.time.attrs.get("units", "")).casefold() != "seconds (unix time)":
            raise ValueError(f"{path.name}: expected one-dimensional Unix-second time")
        fill = float(ds.attrs.get("Fill_value", -9999))
        seconds = ds.time.values.astype(float)
        if not np.isfinite(seconds).all() or np.any(seconds == fill):
            raise ValueError(f"{path.name}: invalid timestamps")
        columns = {}
        for source, column in zip(("x_gse", "y_gse", "z_gse"), POSITIONS):
            variable = ds[source]
            if variable.shape != seconds.shape:
                raise ValueError(f"{path.name}: {source} has incompatible shape")
            if str(variable.attrs.get("units", "")).strip().casefold() not in ("kilometers - km", "km"):
                raise ValueError(f"{path.name}: {source} must be in kilometres")
            values = variable.values.astype(float).copy()
            invalid = ~np.isfinite(values) | (values == fill)
            for attribute in ("_FillValue", "missing_value"):
                for sentinel in np.asarray(variable.attrs.get(attribute, [])).ravel():
                    invalid |= values == sentinel
            for attribute, compare in (("valid_min", np.less), ("valid_max", np.greater)):
                if attribute in variable.attrs:
                    invalid |= compare(values, variable.attrs[attribute])
            if "valid_range" in variable.attrs:
                bounds = variable.attrs["valid_range"]
                invalid |= (values < bounds[0]) | (values > bounds[1])
            values[invalid] = np.nan
            columns[column] = values
        frame = pd.DataFrame(columns, index=pd.DatetimeIndex(
            pd.to_datetime(seconds, unit="s", origin="unix", utc=True), name="time"))
    frame.attrs = {"coordinate_system": "GSE", "units": "km", "source": path.name}
    return frame


def _validate(frame: pd.DataFrame, columns: tuple, kind: str,
              start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans or frame.index.has_duplicates:
        raise ValueError(f"{kind}: expected unique datetime timestamps")
    if any(column not in frame for column in columns):
        raise ValueError(f"{kind}: expected columns {columns}")
    if frame.attrs.get("coordinate_system", "GSE").upper() != "GSE":
        raise ValueError(f"{kind}: coordinates must be GSE")
    expected_units = "nT" if kind == "magnetic" else "km"
    if frame.attrs.get("units", expected_units) != expected_units:
        raise ValueError(f"{kind}: units must be {expected_units}")
    result = frame.loc[:, list(columns)].astype(float).copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index, utc=True), name="time")
    result = result.sort_index().replace([np.inf, -np.inf], np.nan)
    if kind == "magnetic":
        result = result.loc[(result.index >= start) & (result.index < end)]
    else:
        padding = pd.Timedelta(minutes=5)
        result = result.loc[(result.index >= start - padding) & (result.index <= end + padding)]
        if not np.isfinite(result.to_numpy()).all(axis=1).any():
            raise ValueError("positions: no valid XYZ samples")
    if result.empty:
        raise ValueError(f"{kind}: no samples in requested interval")
    result.attrs.update(coordinate_system="GSE", units=expected_units)
    return result


def _save_parquet(frame: pd.DataFrame, target: Path) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
        frame.to_parquet(temporary)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_day(day: date, cache_dir: Path, output_dir: Path, force: bool = False) -> Path:
    """Prepare six pairs, retain successful downloads on failure, then write manifest."""
    start = pd.Timestamp(day, tz="UTC")
    end = start + pd.Timedelta(days=1)
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    local = cache_dir / f"L2_AL1_MAG_{day.strftime('%Y%m%d')}_V00.nc"
    if not local.is_file():
        raise RuntimeError(f"Required local Aditya-L1 file is missing: {local}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, errors, coverage = {}, [], []
    for spacecraft, spec in SPACECRAFT.items():
        entry = {}
        for kind, columns in (("magnetic", MAGNETIC), ("positions", POSITIONS)):
            path = output_dir / f"{spacecraft.lower()}_{kind}.parquet"
            try:
                if path.exists() and not force:
                    log.info("%s %s: reuse %s", spec.label, kind, path)
                    frame = pd.read_parquet(path)
                elif kind == "magnetic":
                    log.info("%s: retrieving native magnetic data", spec.label)
                    frame = fetch_magnetic_field(spacecraft, start, end, cache_dir, force=force).df
                elif spacecraft == "ADITYA-L1":
                    log.info("Aditya-L1: reading local GSE positions")
                    frame = _aditya_positions(local)
                else:
                    log.info("%s: retrieving SSCWeb GSE positions (%s)", spec.label, spec.ssc_id)
                    padding = pd.Timedelta(minutes=5)
                    frame = fetch_ephemeris(spec.ssc_id, start - padding, end + padding, coordinate_system="GSE")
                frame = _validate(frame, columns, kind, start, end)
                _save_parquet(frame, path)
                log.info("%s %s: %d samples, %s through %s", spec.label, kind,
                         len(frame), frame.index[0], frame.index[-1])
                if kind == "positions":
                    valid = frame.dropna()
                    coverage.append((valid.index[0], valid.index[-1]))
                entry[kind] = path.name
            except Exception as error:
                # Preserve all completed products and continue with independent missions.
                errors.append(f"{spec.label} {kind}: {error}")
                log.error("%s", errors[-1])
        manifest[spec.label] = entry
    if errors:
        raise RuntimeError("Inputs incomplete; successful files were retained. Rerun to retry missing inputs.\n"
                           + "\n".join(errors))
    if max(first for first, _ in coverage) > min(last for _, last in coverage):
        raise RuntimeError("The six position products have no shared time coverage")
    path = output_dir / "inputs.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output_dir, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=date(2026, 4, 1), help="UTC day YYYY-MM-DD (default: 2026-04-01)")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"), help="Provider cache and local PRADAN files")
    parser.add_argument("--output-dir", type=Path, help="Prepared tables and manifest (default: cache/constellation/DATE)")
    parser.add_argument("--force", action="store_true", help="Refresh remote/provider caches and reread all prepared inputs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output = args.output_dir or args.cache_dir / "constellation" / args.date.isoformat()
    try:
        manifest = prepare_day(args.date, args.cache_dir, output, args.force)
    except Exception as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Wrote: {manifest}")
    print("Plot with:\n  l1obs constellation --manifest " + shlex.quote(str(manifest))
          + f" --start {args.date.isoformat()}T00:00:00Z --end "
          + (pd.Timestamp(args.date) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
          + " --coordinate-system GSE --magnitude-column b_mag --output "
          + shlex.quote(f"output/constellation_{args.date.isoformat()}.png") + " --pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
