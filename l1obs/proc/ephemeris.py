"""Common-time spacecraft ephemeris processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from l1obs.config import SPACECRAFT, SpacecraftSpec
from l1obs.fetch.sscweb import EphemerisFetchError, fetch_ephemeris


TimeLike = Union[str, datetime, pd.Timestamp]


@dataclass
class SpacecraftConfiguration:
    """Spacecraft positions evaluated at one common UTC epoch."""

    time: datetime
    frame: str
    positions: pd.DataFrame
    errors: Dict[str, str] = field(default_factory=dict)


def _as_utc_timestamp(value: TimeLike) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Target time must be a valid datetime.")
    if timestamp.tz is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _prepare_samples(
    times: Iterable[TimeLike],
    positions: Sequence[Sequence[float]],
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    try:
        sample_times = pd.DatetimeIndex(pd.to_datetime(list(times), utc=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("Ephemeris times must be valid datetimes.") from exc

    try:
        sample_positions = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Ephemeris positions must be numeric XYZ values.") from exc

    if sample_positions.ndim != 2 or sample_positions.shape[1] != 3:
        raise ValueError("Ephemeris positions must have shape (n_samples, 3).")
    if len(sample_times) != len(sample_positions):
        raise ValueError("Ephemeris times and positions must have the same length.")
    if len(sample_times) == 0:
        raise ValueError("At least one ephemeris sample is required.")
    if sample_times.hasnans:
        raise ValueError("Ephemeris times cannot contain missing values.")
    if not np.isfinite(sample_positions).all():
        raise ValueError("Ephemeris positions must contain only finite values.")

    order = np.argsort(sample_times.asi8, kind="stable")
    sample_times = sample_times[order]
    sample_positions = sample_positions[order]
    if sample_times.duplicated().any():
        raise ValueError("Ephemeris times must be unique.")
    return sample_times, sample_positions


def _bracketing_indices(
    times: pd.DatetimeIndex,
    target_time: pd.Timestamp,
) -> Tuple[int, int]:
    target_ns = target_time.value
    sample_ns = times.asi8
    insertion = int(np.searchsorted(sample_ns, target_ns, side="left"))

    if insertion < len(times) and sample_ns[insertion] == target_ns:
        return insertion, insertion
    if insertion == 0 or insertion == len(times):
        raise ValueError(
            f"Target time {target_time.isoformat()} is not bracketed by ephemeris samples."
        )
    return insertion - 1, insertion


def interpolate_position(
    times: Iterable[TimeLike],
    positions: Sequence[Sequence[float]],
    target_time: TimeLike,
) -> np.ndarray:
    """Linearly interpolate an XYZ position to a bracketed target time.

    An exact timestamp returns its position directly. Otherwise samples must
    exist strictly before and after the requested UTC epoch.
    """
    sample_times, sample_positions = _prepare_samples(times, positions)
    target = _as_utc_timestamp(target_time)
    before, after = _bracketing_indices(sample_times, target)

    if before == after:
        return sample_positions[before].copy()

    before_ns = sample_times[before].value
    after_ns = sample_times[after].value
    fraction = (target.value - before_ns) / (after_ns - before_ns)
    return sample_positions[before] + fraction * (
        sample_positions[after] - sample_positions[before]
    )


def _resolve_spacecraft(
    requested: Optional[Sequence[str]],
) -> Sequence[Tuple[str, SpacecraftSpec]]:
    if requested is None:
        return list(SPACECRAFT.items())
    if isinstance(requested, str):
        requested = [requested]

    lookup = {}
    for key, spec in SPACECRAFT.items():
        lookup[key.casefold()] = key
        lookup[spec.label.casefold()] = key

    resolved = []
    seen = set()
    for name in requested:
        key = lookup.get(name.casefold())
        if key is None:
            choices = ", ".join(spec.label for spec in SPACECRAFT.values())
            raise ValueError(f"Unknown spacecraft {name!r}; expected one of: {choices}.")
        if key not in seen:
            resolved.append((key, SPACECRAFT[key]))
            seen.add(key)
    if not resolved:
        raise ValueError("At least one spacecraft must be requested.")
    return resolved


def get_positions_at_time(
    target_time: TimeLike,
    spacecraft: Optional[Sequence[str]] = None,
    coordinate_system: str = "GSE",
    window_minutes: float = 30,
) -> SpacecraftConfiguration:
    """Fetch and interpolate spacecraft positions to one common UTC epoch."""
    target = _as_utc_timestamp(target_time)
    if window_minutes <= 0:
        raise ValueError("window_minutes must be greater than zero.")

    requested = _resolve_spacecraft(spacecraft)
    delta = pd.Timedelta(minutes=window_minutes)
    rows = []
    errors = {}

    for _, spec in requested:
        try:
            ephemeris = fetch_ephemeris(
                spec.ssc_id,
                target - delta,
                target + delta,
                coordinate_system=coordinate_system,
            )
            sample_times, sample_positions = _prepare_samples(
                ephemeris.index,
                ephemeris[["x_km", "y_km", "z_km"]].to_numpy(),
            )
            before_index, after_index = _bracketing_indices(sample_times, target)
            position = interpolate_position(sample_times, sample_positions, target)
            rows.append(
                {
                    "spacecraft": spec.label,
                    "x_km": position[0],
                    "y_km": position[1],
                    "z_km": position[2],
                    "sample_before": sample_times[before_index],
                    "sample_after": sample_times[after_index],
                }
            )
        except (EphemerisFetchError, ValueError) as exc:
            errors[spec.label] = str(exc)

    if not rows:
        details = "; ".join(f"{name}: {message}" for name, message in errors.items())
        raise RuntimeError(
            f"No spacecraft positions are available at {target.isoformat()}. {details}"
        )

    positions = pd.DataFrame(
        rows,
        columns=[
            "spacecraft",
            "x_km",
            "y_km",
            "z_km",
            "sample_before",
            "sample_after",
        ],
    )
    return SpacecraftConfiguration(
        time=target.to_pydatetime(),
        frame=coordinate_system.upper(),
        positions=positions,
        errors=errors,
    )
