"""Crossing-epoch GSE positions with explicit bracketing and gap limits."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from l1obs.proc.ephemeris import interpolate_position
from .data import R_COLUMNS, prepare_table
from .event import positive, utc

log = logging.getLogger(__name__)


def position_at(ephemeris: pd.DataFrame, time, max_gap_s: float = 900) -> tuple[np.ndarray, dict]:
    """Interpolate consecutive finite samples, never across NaNs or large gaps.

    Exact finite samples are allowed without interpolation, including endpoints.
    The gap threshold is explicit, not inferred from a potentially sparse table.
    """
    limit = positive(max_gap_s, "ephemeris max_gap_s")
    frame = prepare_table(ephemeris, R_COLUMNS, "km", "ephemeris")
    return _position_at_prepared(frame, utc(time), limit)


def _position_at_prepared(frame, target, limit):
    # Internal fast path for Monte Carlo; caller prepares the table once.
    times = frame.index
    insertion = int(times.searchsorted(target))
    if insertion < len(times) and times[insertion] == target:
        before = after = insertion
    elif insertion == 0 or insertion == len(times):
        raise ValueError(f"{target.isoformat()}: not bracketed by ephemeris; extrapolation forbidden")
    else:
        before, after = insertion - 1, insertion
    samples = frame.iloc[before:after + 1][list(R_COLUMNS)].to_numpy(float)
    gap = (times[after] - times[before]).total_seconds()
    if not np.isfinite(samples).all():
        raise ValueError(f"{target.isoformat()}: invalid bracketing ephemeris position")
    if gap > limit:
        raise ValueError(f"{target.isoformat()}: ephemeris gap {gap:g}s exceeds {limit:g}s")
    # Reuse the existing numerical interpolator on just the bracket.
    vector = interpolate_position(times[before:after + 1], samples, target)
    metadata = {"time": target.isoformat(), "sample_before": times[before].isoformat(),
                "sample_after": times[after].isoformat(), "bracket_gap_s": gap,
                "method": "exact sample" if before == after else "linear in UTC between consecutive finite samples",
                "max_gap_s": limit, "frame": "GSE", "units": "km"}
    log.debug("Position interpolation: %s", metadata)
    return vector, metadata


def positions_at_arrivals(ephemerides: dict, times: dict, max_gap_s: float = 900) -> tuple[np.ndarray, dict]:
    positions, metadata = [], {}
    for name, time in times.items():
        if name not in ephemerides:
            raise ValueError(f"Missing ephemeris for {name}")
        vector, info = position_at(ephemerides[name], time, max_gap_s)
        positions.append(vector)
        metadata[name] = info
        log.info("%s position at %s: %s km (%s)", name, info["time"], vector, info["method"])
    return np.asarray(positions), metadata


def gse_rotation_estimate(positions_km, span_s: float, speed_km_s: float) -> dict:
    """Order-of-magnitude annual axis rotation bound; not an applied correction.

    GSE's approximately annual solar direction rotation is used here. This is
    neither a full precession/nutation model nor a position-error covariance.
    Earth-centered radius gives a conservative displacement scale.
    """
    omega = 2 * np.pi / (365.25 * 86400)
    angle = omega * span_s
    displacement = float(np.max(np.linalg.norm(positions_km, axis=1)) * angle)
    return {"approximate_angular_rate_rad_s": omega, "crossing_span_s": span_s,
            "axis_rotation_rad": angle, "displacement_scale_km": displacement,
            "timing_scale_s": displacement / speed_km_s,
            "applied_correction": False,
            "interpretation": "conservative annual-rotation scale using Earth-centered radius; not a full frame transform"}
