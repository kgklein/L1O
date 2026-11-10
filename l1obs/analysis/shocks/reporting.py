"""Strict JSON/CSV serialization with software and scientific provenance."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [json_value(v) for v in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return value.isoformat() if isinstance(value, pd.Timestamp) else str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload):
    """Atomic strict JSON; no nonstandard NaN/Infinity tokens."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(json_value(payload), stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    log.info("Wrote %s", path)


def write_csv(path: Path, table):
    frame = table.copy() if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    for column in frame:
        frame[column] = frame[column].map(lambda value: json.dumps(json_value(value), allow_nan=False)
                                         if isinstance(value, (dict, list, tuple)) else value)
    frame.to_csv(path, index=False)
    log.info("Wrote %s", path)


def provenance(event, data):
    versions = {"python": platform.python_version()}
    for name in ("l1_observatory", "numpy", "pandas", "matplotlib", "pyarrow"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    revision = None
    dirty = None
    try:
        root = Path(__file__).resolve().parents[3]
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        pass
    # Include uncommitted/new implementation files in reproducibility metadata.
    package = Path(__file__).resolve().parents[2]
    source_hashes = {str(p.relative_to(package.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(package.rglob("*.py"))}
    return {"source_code_sha256": source_hashes, "event": event.raw, "event_path": str(event.path),
            "event_file_sha256": hashlib.sha256(event.path.read_bytes()).hexdigest() if event.path else None,
            "inputs": data.provenance, "software_versions": versions,
            "git_revision": revision, "git_tracked_changes": dirty,
            "coordinate_system": "GSE", "units": {"positions": "km", "magnetic": "nT", "time": "s", "speed": "km/s"}}


def fit_payload(fit):
    payload = asdict(fit)
    payload["parameter_covariance_order"] = ["offset_s", "qx_s_km", "qy_s_km", "qz_s_km"]
    payload["status"] = "flagged" if any("Large residuals" in f or "Weak geometric" in f or "exceeds 3" in f for f in fit.flags) else "computed"
    return payload


def residual_table(fit):
    return pd.DataFrame({"spacecraft": fit.spacecraft, "observed_time": fit.observed_times,
                         "predicted_time": fit.predicted_times, "residual_s": fit.residuals_s,
                         "uncertainty_s": np.nan if fit.uncertainties_s is None else fit.uncertainties_s,
                         "observed_offset_s": fit.observed_offsets_s, "predicted_offset_s": fit.predicted_offsets_s})
