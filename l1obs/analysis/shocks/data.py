"""Native local inputs using the shared constellation manifest and schema.

Archive fill values and quality flags must already have been normalized by the
existing mission loaders. This module never resamples or fills magnetic data.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from l1obs.config import DATASETS
from l1obs.io.manifest import load_manifest
from .event import Event, positive, spacecraft_key

log = logging.getLogger(__name__)
B_COLUMNS = ("bx_gse", "by_gse", "bz_gse")
R_COLUMNS = ("x_km", "y_km", "z_km")


@dataclass
class EventData:
    magnetic: dict[str, pd.DataFrame]
    ephemerides: dict[str, pd.DataFrame]
    provenance: dict


def prepare_table(frame: pd.DataFrame, columns: tuple, units: str, label: str) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans or frame.index.has_duplicates:
        raise ValueError(f"{label}: requires unique datetime timestamps without NaT")
    if frame.empty or any(c not in frame for c in columns):
        raise ValueError(f"{label}: empty table or missing columns {columns}")
    for key in ("frame", "coordinate_system"):
        if str(frame.attrs.get(key, "GSE")).upper() != "GSE":
            raise ValueError(f"{label}: requires GSE coordinates")
    metadata = frame.attrs.get("units", units)
    declared = [metadata.get(c, units) for c in columns] if isinstance(metadata, dict) else [metadata]
    if any(str(u).casefold() != units.casefold() for u in declared):
        raise ValueError(f"{label}: expected units {units}")
    result = frame.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index, utc=True), name="time").as_unit("ns")
    for c in columns:
        result[c] = pd.to_numeric(result[c], errors="raise")
    return result.sort_index().replace([np.inf, -np.inf], np.nan)


def native_cadence(frame: pd.DataFrame) -> float:
    if len(frame) < 2:
        raise ValueError("At least two samples are needed to determine native cadence")
    return positive(np.median(np.diff(frame.index.as_unit("ns").asi8)) / 1e9, "native cadence")


def load_event_data(event: Event, manifest: str | Path | None = None) -> EventData:
    """Load existing CSV/Parquet pairs, retaining native timestamps and NaN rows.

    The event must explicitly assert the local tables are native, quality-masked
    GSE/nT/km data. Metadata contradictions are errors. Files remain unchanged.
    To retrieve new files use fetch_magnetic_field/fetch_ephemeris or the existing
    scripts/prepare_constellation.py before this fully offline analysis.
    """
    inputs = event.inputs
    if inputs.get("native_cadence") is not True or inputs.get("quality_masks_applied") is not True:
        raise ValueError("inputs must explicitly assert native_cadence=true and quality_masks_applied=true")
    if inputs.get("coordinate_system") != "GSE" or inputs.get("magnetic_units") != "nT" or inputs.get("position_units") != "km":
        raise ValueError("inputs must declare GSE, magnetic_units=nT, position_units=km")
    filename = manifest or inputs.get("manifest")
    if filename is None:
        raise ValueError("Supply a local manifest via --manifest or inputs.manifest")
    path = Path(filename).resolve()
    loaded = load_manifest(path, expected_count=None)
    canonical = {}
    for name, pair in loaded.items():
        key = spacecraft_key(name)
        if key in canonical:
            raise ValueError(f"Duplicate spacecraft alias in manifest: {name}")
        canonical[key] = pair
    missing = set(event.spacecraft) - canonical.keys()
    if missing:
        raise ValueError(f"Manifest missing spacecraft: {sorted(missing)}")
    magnetic, ephemerides, provenance = {}, {}, {}
    entries = {spacecraft_key(n): v for n, v in json.loads(path.read_text()).items()}
    for name in event.spacecraft:
        pair = canonical[name]
        raw = pair["magnetic"]
        if raw.attrs.get("resampled") or raw.attrs.get("native_cadence") is False:
            raise ValueError(f"{name}: resampled magnetic data are forbidden for shock timing")
        frame = prepare_table(raw, B_COLUMNS, "nT", f"{name} magnetic")
        if "b_mag" in frame:
            frame["b_mag"] = pd.to_numeric(frame["b_mag"], errors="raise")
            magnitude_source = raw.attrs.get("magnitude_source", "input b_mag (not filled from components)")
        else:
            frame["b_mag"] = np.sqrt(np.sum(frame[list(B_COLUMNS)].to_numpy() ** 2, axis=1))
            magnitude_source = "calculated GSE vector norm; missing component => missing magnitude"
        frame.loc[frame.b_mag < 0, "b_mag"] = np.nan
        frame = frame.loc[(frame.index >= event.start) & (frame.index <= event.end)].copy()
        if frame.empty:
            raise ValueError(f"{name}: no magnetic measurements in event interval")
        cadence = native_cadence(frame)
        frame.attrs.update(spacecraft=name, magnitude_source=magnitude_source, coordinate_system="GSE", units="nT")
        magnetic[name] = frame
        ephemerides[name] = prepare_table(pair["positions"], R_COLUMNS, "km", f"{name} ephemeris")
        sources = {}
        for kind, filename in entries[name].items():
            source = (path.parent / filename).resolve()
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash(stream)
            sources[kind] = {"path": str(source), "sha256": digest,
                             "metadata": dict(pair[kind].attrs)}
            log.info("%s %s input: %s", name, kind, source)
        provenance[name] = {"files": sources, "configured_dataset": DATASETS[name]["mag"],
                            "dataset_identity_verified": False,
                            "native_cadence_s": cadence, "magnitude_source": magnitude_source,
                            "user_metadata": inputs.get("spacecraft_metadata", {}).get(name, {})}
    return EventData(magnetic, ephemerides, {"manifest": str(path), "spacecraft": provenance})


def _hash(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()
