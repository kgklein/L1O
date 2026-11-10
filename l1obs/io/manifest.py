"""Shared local constellation tables; no retrieval or time alignment."""
from pathlib import Path
import json

import pandas as pd


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key!r}")
        result[key] = value
    return result


def load_manifest(manifest_path: Path, expected_count: int | None = 6) -> dict:
    """Load local table pairs, resolving paths relative to the manifest.

    The default six-member constraint preserves the constellation CLI contract;
    analysis callers may pass expected_count=None to support explicit subsets.
    """
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(), object_pairs_hook=_unique_json_object)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read constellation manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or not manifest or (expected_count is not None and len(manifest) != expected_count):
        raise RuntimeError("The constellation manifest must map exactly six spacecraft names to file pairs.")
    datasets = {}
    for name, entry in manifest.items():
        if not name.strip() or not isinstance(entry, dict) or set(entry) != {"magnetic", "positions"}:
            raise RuntimeError(f"Manifest entry {name!r} requires magnetic and positions file paths.")
        datasets[name] = {}
        for kind, filename in entry.items():
            if not isinstance(filename, str) or not filename.strip():
                raise RuntimeError(f"{name} {kind} file path must be a nonempty string.")
            path = Path(filename)
            if not path.is_absolute():
                path = manifest_path.parent / path
            try:
                if path.suffix.lower() == ".parquet":
                    frame = pd.read_parquet(path)
                elif path.suffix.lower() == ".csv":
                    frame = pd.read_csv(path)
                else:
                    raise ValueError("supported file extensions are .parquet and .csv")
                if "time" in frame.columns:
                    time = pd.to_datetime(frame.pop("time"), utc=True, format="mixed")
                    frame.index = pd.DatetimeIndex(time, name="time")
                elif not isinstance(frame.index, pd.DatetimeIndex):
                    raise ValueError("table requires a datetime index or a time column")
                datasets[name][kind] = frame
            except (OSError, ValueError, TypeError, ImportError) as exc:
                raise RuntimeError(f"Cannot load {name} {kind} from {path}: {exc}") from exc
    return datasets


