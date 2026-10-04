"""Mission-level access to native-cadence magnetic measurements."""
from pathlib import Path

import pandas as pd

from l1obs.config import DATASETS
from .cdaweb import fetch_cdaweb_dataset
from .models import FetchResult
from .ncei import fetch_ncei_product


def fetch_magnetic_field(spacecraft: str, start: pd.Timestamp, end: pd.Timestamp,
                         cache_dir: Path, force: bool = False) -> FetchResult:
    """Fetch normalized GSE components and archive magnitude without resampling."""
    name = spacecraft.strip().upper()
    spec = DATASETS.get(name, {})
    if "mag" not in spec:
        raise ValueError(f"No magnetic product configured for {spacecraft!r}")
    provider = spec.get("mag_provider", "cdaweb")
    if provider == "cdaweb":
        return fetch_cdaweb_dataset(spec["mag"], start, end, cache_dir, force=force)
    if provider == "ncei":
        return fetch_ncei_product(spec["mag"], start, end, cache_dir, force=force)
    raise ValueError(f"Unknown magnetic provider {provider!r} for {spacecraft!r}")
