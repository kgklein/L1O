"""Validated, versionable shock event definitions. Naive input dates mean UTC."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from l1obs.config import SPACECRAFT
from l1obs.io.manifest import _unique_json_object


def utc(value) -> pd.Timestamp:
    if value is None:
        raise ValueError("A UTC timestamp is required")
    time = pd.Timestamp(str(value) if isinstance(value, (str, np.str_)) else value)
    if pd.isna(time):
        raise ValueError("A UTC timestamp cannot be NaT")
    return time.tz_localize("UTC") if time.tz is None else time.tz_convert("UTC")


def positive(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean")
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def spacecraft_key(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("Spacecraft identity must be a string")
    for key, spec in SPACECRAFT.items():
        if name.casefold() in (key.casefold(), spec.label.casefold()):
            return key
    raise ValueError(f"Unknown spacecraft: {name!r}")


@dataclass(frozen=True)
class Event:
    event_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    spacecraft: tuple[str, ...]
    output_dir: Path
    search_windows: dict = field(default_factory=dict)
    arrivals: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    detection: dict = field(default_factory=dict)
    fit: dict = field(default_factory=dict)
    monte_carlo: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)
    path: Path | None = None

    def window(self, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.search_windows.get(name, (self.start, self.end))


def load_event(path: str | Path) -> Event:
    """Load an event without requiring any accepted arrivals or available files.

    Manifest and output paths resolve relative to the event JSON, not the cwd.
    Only explicitly accepted manual selections/validations can enter a fit.
    """
    path = Path(path).resolve()
    data = json.loads(path.read_text(), object_pairs_hook=_unique_json_object)
    if not isinstance(data, dict):
        raise ValueError("Event must be a JSON object")
    allowed = set(Event.__dataclass_fields__) - {"raw", "path"}
    if set(data) - allowed:
        raise ValueError(f"Unknown event fields: {sorted(set(data) - allowed)}")
    identifier = data.get("event_id", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier):
        raise ValueError("event_id must be a safe nonempty filename identifier")
    start, end = utc(data.get("start")), utc(data.get("end"))
    if end <= start:
        raise ValueError("Event end must follow start")
    if not isinstance(data.get("spacecraft", []), list):
        raise ValueError("spacecraft must be a list of identities")
    names = tuple(spacecraft_key(n) for n in data.get("spacecraft", []))
    if not names or len(set(names)) != len(names):
        raise ValueError("Event requires unique spacecraft")
    sections = {}
    for section in ("search_windows", "arrivals"):
        value = data.get(section, {})
        if not isinstance(value, dict):
            raise ValueError(f"{section} must be an object")
        normalized = {}
        for name, entry in value.items():
            key = spacecraft_key(name)
            if key not in names or key in normalized:
                raise ValueError(f"Unknown or duplicate {section} spacecraft {name}")
            normalized[key] = entry
        sections[section] = normalized
    for name, window in sections["search_windows"].items():
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError(f"{name}: search window must be [start, end]")
        lo, hi = map(utc, window)
        if not start <= lo < hi <= end:
            raise ValueError(f"{name}: search window must lie inside event interval")
        sections["search_windows"][name] = (lo, hi)
    for name, arrival in sections["arrivals"].items():
        if not isinstance(arrival, dict):
            raise ValueError(f"{name}: arrival must be an object")
        unknown = set(arrival) - {"time", "uncertainty_s", "accepted", "method", "notes",
                                  "ramp_duration_s", "precision_justification"}
        if unknown:
            raise ValueError(f"{name}: unknown arrival fields {sorted(unknown)}")
        if "accepted" in arrival and not isinstance(arrival["accepted"], bool):
            raise ValueError(f"{name}: accepted must be a JSON boolean")
        if arrival.get("accepted", False) and arrival.get("method") not in ("manual_selected", "manual_validated"):
            raise ValueError(f"{name}: accepted arrivals require a manual selection/validation method")
        if arrival.get("time") is not None:
            time = utc(arrival["time"])
            if not start <= time <= end:
                raise ValueError(f"{name}: arrival outside inspection interval")
        elif arrival.get("accepted", False):
            raise ValueError(f"{name}: accepted arrival requires time")
        if arrival.get("uncertainty_s") is not None:
            positive(arrival["uncertainty_s"], "uncertainty_s")
        for text_field in ("notes", "precision_justification"):
            if text_field in arrival and not isinstance(arrival[text_field], str):
                raise ValueError(f"{name}: {text_field} must be a string")
        if arrival.get("ramp_duration_s") is not None:
            positive(arrival["ramp_duration_s"], "ramp_duration_s")
    options = {}
    for section in ("inputs", "detection", "fit", "monte_carlo"):
        value = data.get(section, {})
        if not isinstance(value, dict):
            raise ValueError(f"{section} must be an object")
        options[section] = value.copy()
    allowed_options = {
        "inputs": {"manifest", "native_cadence", "quality_masks_applied", "coordinate_system", "magnetic_units",
                   "position_units", "ephemeris_max_gap_s", "spacecraft_metadata"},
        "detection": {"window_s", "min_jump_nt", "min_significance", "noise_floor_nt", "ambiguity_ratio", "max_candidates", "max_gap_s"},
        "fit": {"method", "condition_limit", "covariance_s2"},
        "monte_carlo": {"samples", "seed", "recompute_positions"},
    }
    for section, allowed_keys in allowed_options.items():
        unknown = set(options[section]) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown {section} options: {sorted(unknown)}")
    if "recompute_positions" in options["monte_carlo"] and not isinstance(options["monte_carlo"]["recompute_positions"], bool):
        raise ValueError("monte_carlo.recompute_positions must be a JSON boolean")
    inputs = options["inputs"]
    if "ephemeris_max_gap_s" in inputs:
        positive(inputs["ephemeris_max_gap_s"], "ephemeris_max_gap_s")
    if inputs.get("manifest"):
        inputs["manifest"] = str((path.parent / inputs["manifest"]).resolve())
    if "spacecraft_metadata" in inputs:
        metadata = inputs["spacecraft_metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("inputs.spacecraft_metadata must be an object")
        normalized = {}
        for name, entry in metadata.items():
            key = spacecraft_key(name)
            if key in normalized or key not in names or not isinstance(entry, dict):
                raise ValueError(f"Invalid/duplicate spacecraft_metadata entry {name}")
            unknown = set(entry) - {"timestamp_convention", "timestamp_offset_s", "timestamp_uncertainty_s", "notes"}
            if unknown:
                raise ValueError(f"{name}: unknown spacecraft metadata {sorted(unknown)}")
            for quantity in ("timestamp_offset_s", "timestamp_uncertainty_s"):
                if quantity in entry and not np.isfinite(float(entry[quantity])):
                    raise ValueError(f"{name}: {quantity} must be finite")
            if float(entry.get("timestamp_uncertainty_s", 0)) < 0:
                raise ValueError(f"{name}: timestamp_uncertainty_s must be nonnegative")
            normalized[key] = entry.copy()
        inputs["spacecraft_metadata"] = normalized
    output = path.parent / data.get("output_dir", f"output/shocks/{identifier}")
    return Event(identifier, start, end, names, output.resolve(), **sections, **options,
                 raw=data, path=path)
