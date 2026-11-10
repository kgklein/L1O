"""Robust magnetic enhancement proposals and explicit manual acceptance.

All interpolated ramp thresholds are restricted to consecutive finite native
samples. Detection is a proposal mechanism, not a shock classifier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import logging

import numpy as np
import pandas as pd

from .data import native_cadence
from .event import Event, positive, utc

log = logging.getLogger(__name__)
TIMESTAMP_NOTES = {
    "WIND": "Epoch3 archive epoch; bin reference/clock accuracy not verified here",
    "ACE": "Epoch archive epoch; bin reference/clock accuracy not verified here",
    "DSCOVR": "Epoch1 archive epoch; bin reference/clock accuracy not verified here",
    "ADITYA-L1": "Unix UTC seconds; 10 s averaged bins; start/center convention not established",
    "IMAP": "epoch archive epoch; bin reference/clock accuracy not verified here",
    "SOLAR-1": "time_sec UTC; start of 1 s averaging window (source NetCDF comments)",
}


@dataclass(frozen=True)
class DetectionOptions:
    window_s: float = 60
    min_jump_nt: float = .5
    min_significance: float = 3
    noise_floor_nt: float = .05
    ambiguity_ratio: float = .8
    max_candidates: int = 5
    max_gap_s: float | None = None

    def __post_init__(self):
        for key in ("window_s", "min_jump_nt", "min_significance", "noise_floor_nt"):
            positive(getattr(self, key), key)
        if not 0 < self.ambiguity_ratio <= 1:
            raise ValueError("ambiguity_ratio must lie in (0, 1]")
        if isinstance(self.max_candidates, bool) or int(self.max_candidates) != self.max_candidates or self.max_candidates < 2:
            raise ValueError("max_candidates must be an integer >= 2")
        if self.max_gap_s is not None:
            positive(self.max_gap_s, "max_gap_s")


@dataclass
class DetectionResult:
    status: str
    candidates: list[dict]
    native_cadence_s: float
    notes: str


def _runs(values, times_s, limit):
    valid = np.isfinite(values)
    starts, stops = [], []
    active = None
    for i in range(len(values)):
        if active is not None and (not valid[i] or times_s[i] - times_s[i - 1] > limit):
            starts.append(active)
            stops.append(i)
            active = None
        if valid[i] and active is None:
            active = i
    if active is not None:
        starts.append(active)
        stops.append(len(values))
    return zip(starts, stops)


def _threshold_time(times, values, level, near):
    changes = np.flatnonzero((values[:-1] < level) & (values[1:] >= level))
    if not len(changes):
        return None
    k = changes[np.argmin(np.abs((times[changes] + times[changes + 1]) / 2 - near))]
    return float(times[k] + (level - values[k]) / (values[k + 1] - values[k]) * (times[k + 1] - times[k]))


def detect_arrivals(frame: pd.DataFrame, window=None, options: DetectionOptions | None = None,
                    *, timestamp_offset_s: float = 0, timestamp_uncertainty_s: float = 0) -> DetectionResult:
    """Compare time-window upstream/downstream medians in each valid run.

    Rank positive jumps by amplitude, break ties by the adjacent positive slope,
    and suppress proposals within one comparison-window duration. The ramp
    convention is its 50% level; 10–90% duration is measured on the same run.
    Proposal sigma is max(cadence/2, ramp_duration/2, timestamp_uncertainty_s).
    Timestamp offsets apply only to proposed physical crossing times; the
    original measurements are never shifted. Errors are interpreted as 1 sigma.
    """
    options = options or DetectionOptions()
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Detection requires unique, sorted datetime timestamps")
    frame = frame.copy(deep=False)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).as_unit("ns")
    cadence = native_cadence(frame)
    if not np.isfinite(timestamp_offset_s) or not np.isfinite(timestamp_uncertainty_s) or timestamp_uncertainty_s < 0:
        raise ValueError("Timestamp offset/uncertainty must be finite and uncertainty nonnegative")
    data = frame if window is None else frame.loc[(frame.index >= utc(window[0])) & (frame.index <= utc(window[1]))]
    if data.empty:
        return DetectionResult("missing", [], cadence, "No measurements in search window")
    epoch = data.index[0]
    times = (data.index - epoch).total_seconds().to_numpy()
    values = data.b_mag.to_numpy(float)
    limit = options.max_gap_s or 3 * cadence
    scored = []
    for start, stop in _runs(values, times, limit):
        t, b = times[start:stop], values[start:stop]
        for k in range(1, len(t)):
            midpoint = (t[k - 1] + t[k]) / 2
            left = np.searchsorted(t, midpoint - options.window_s)
            right = np.searchsorted(t, midpoint + options.window_s, side="right")
            up, down = b[left:k], b[k:right]
            # At least three measurements and broad time support on both sides.
            if len(up) < 3 or len(down) < 3:
                continue
            if midpoint - t[left] < .6 * options.window_s or t[right - 1] - midpoint < .6 * options.window_s:
                continue
            upstream, downstream = np.median(up), np.median(down)
            jump = downstream - upstream
            noise = max(1.4826 * np.median(np.abs(up - upstream)),
                        1.4826 * np.median(np.abs(down - downstream)), options.noise_floor_nt)
            significance = jump / noise
            if jump < options.min_jump_nt or significance < options.min_significance:
                continue
            slope = (b[k] - b[k - 1]) / (t[k] - t[k - 1])
            scored.append((float(jump), float(slope), float(significance), midpoint,
                           t[left:right], b[left:right], float(upstream), float(downstream)))
    scored.sort(key=lambda s: (s[0], s[1], s[2]), reverse=True)
    candidates = []
    for jump, slope, significance, midpoint, t, b, upstream, downstream in scored:
        ramp50 = _threshold_time(t, b, upstream + .5 * jump, midpoint)
        if ramp50 is None or any(abs(ramp50 - c["raw_relative_time_s"]) < options.window_s for c in candidates):
            continue
        ramp10 = _threshold_time(t, b, upstream + .1 * jump, ramp50)
        ramp90 = _threshold_time(t, b, upstream + .9 * jump, ramp50)
        resolved = ramp10 is not None and ramp90 is not None and ramp10 <= ramp50 <= ramp90
        bracket_index = int(np.clip(np.searchsorted(t, ramp50), 1, len(t) - 1))
        local_spacing = float(t[bracket_index] - t[bracket_index - 1])
        duration = max(cadence, local_spacing, ramp90 - ramp10) if resolved else options.window_s
        time = epoch + pd.Timedelta(seconds=ramp50 + timestamp_offset_s)
        candidates.append({"crossing_time": time.round("ms").isoformat(), "uncertainty_s": max(cadence / 2, duration / 2, timestamp_uncertainty_s),
                           "ramp_duration_s": duration, "jump_nt": jump, "significance": significance,
                           "upstream_median_nt": upstream, "downstream_median_nt": downstream,
                           "raw_relative_time_s": ramp50, "native_cadence_s": cadence, "local_sample_spacing_s": local_spacing,
                           "timestamp_offset_s": timestamp_offset_s, "accepted": False,
                           "method": "automatically_proposed", "quality": "proposed" if resolved else "complex_ramp"})
        if len(candidates) >= options.max_candidates:
            break
    if not candidates:
        return DetectionResult("no_candidate", [], cadence, "No positive discontinuity with contiguous support")
    ambiguous = len(candidates) > 1 and candidates[1]["jump_nt"] >= options.ambiguity_ratio * candidates[0]["jump_nt"]
    status = "ambiguous" if ambiguous or candidates[0]["quality"] == "complex_ramp" else "proposed"
    return DetectionResult(status, candidates, cadence,
                           "Manual validation required; detector uses archive sample epochs plus explicit timestamp_offset_s")


def validate_magnetic_support(frame, crossing_time, max_gap_s=None):
    """Reject a selected time outside coverage, at NaNs, or spanning a gap."""
    frame = frame.copy(deep=False)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).as_unit("ns")
    target = utc(crossing_time)
    k = int(frame.index.searchsorted(target))
    if k < len(frame) and frame.index[k] == target:
        bracket = frame.iloc[k:k + 1]
    elif k == 0 or k == len(frame):
        raise ValueError("Crossing time is outside native magnetic coverage")
    else:
        bracket = frame.iloc[k - 1:k + 1]
    if not np.isfinite(bracket.b_mag).all():
        raise ValueError("Crossing time spans invalid magnetic measurements")
    spacing = native_cadence(frame) if len(bracket) == 1 else (bracket.index[-1] - bracket.index[0]).total_seconds()
    if len(bracket) == 2 and spacing > (max_gap_s or 3 * native_cadence(frame)):
        raise ValueError("Crossing time spans a magnetic timestamp gap")
    return spacing


def build_arrival_table(event: Event, magnetic: dict) -> tuple[pd.DataFrame, dict]:
    """Suggest crossings, retaining any configured manual values verbatim.

    A fit must separately require accepted rows. Never writes the event file.
    Missing manual uncertainties are allowed for explicit unweighted fits.
    """
    options = DetectionOptions(**event.detection)
    rows, detections = [], {}
    for name in event.spacecraft:
        frame = magnetic[name]
        metadata = event.inputs.get("spacecraft_metadata", {}).get(name, {})
        offset = float(metadata.get("timestamp_offset_s", 0))
        clock_sigma = float(metadata.get("timestamp_uncertainty_s", 0))
        detection = detect_arrivals(frame, event.window(name), options, timestamp_offset_s=offset,
                                    timestamp_uncertainty_s=clock_sigma)
        detections[name] = asdict(detection)
        log.info("%s detection %s: %s", name, detection.status, detection.candidates)
        manual = event.arrivals.get(name, {})
        row = {"spacecraft": name, "crossing_time": None, "uncertainty_s": None, "accepted": False,
               "method": "unselected", "native_cadence_s": detection.native_cadence_s,
               "ramp_duration_s": None, "quality": detection.status, "notes": "",
               "timestamp_convention": metadata.get("timestamp_convention", TIMESTAMP_NOTES[name]),
               "proposal_timestamp_offset_s": offset, "precision_justification": ""}
        if manual.get("time") is not None:
            time = utc(manual["time"])
            # Manual times denote physical crossing epochs, not shifted sample labels.
            local_spacing = validate_magnetic_support(frame, time - pd.Timedelta(seconds=offset), options.max_gap_s)
            sigma = manual.get("uncertainty_s")
            justification = manual.get("precision_justification", "")
            if sigma is not None and float(sigma) < max(detection.native_cadence_s / 2, local_spacing / 2, clock_sigma) and not justification.strip():
                raise ValueError(f"{name}: sub-cadence/clock-floor uncertainty requires precision_justification")
            row.update(crossing_time=time.isoformat(), uncertainty_s=sigma,
                       accepted=manual.get("accepted", False), method=manual.get("method", "manual_selected"),
                       ramp_duration_s=manual.get("ramp_duration_s"), quality="accepted" if manual.get("accepted") else "unaccepted_manual",
                       notes=manual.get("notes", ""), precision_justification=justification)
            log.info("%s manual crossing: %s (accepted=%s)", name, time, row["accepted"])
        elif detection.candidates:
            best = detection.candidates[0]
            row.update({k: best[k] for k in ("crossing_time", "uncertainty_s", "method", "ramp_duration_s")})
            row["notes"] = detection.notes
        rows.append(row)
    return pd.DataFrame(rows), detections


def accepted_arrivals(table: pd.DataFrame, spacecraft=None) -> pd.DataFrame:
    """Require explicitly accepted manual arrivals for every requested member."""
    selected = table if spacecraft is None else table.loc[table.spacecraft.isin(spacecraft)]
    if spacecraft is not None and set(selected.spacecraft) != set(spacecraft):
        raise ValueError("Missing requested arrival measurements")
    if len(selected) < 4:
        raise ValueError("At least four accepted spacecraft arrivals are required")
    invalid = selected.loc[~selected.accepted.eq(True) | ~selected.method.isin(["manual_selected", "manual_validated"])]
    if len(invalid) or selected.crossing_time.isna().any():
        raise ValueError("Explicitly accepted manual arrivals required for all fit spacecraft: " + ", ".join(invalid.spacecraft))
    return selected.copy()
