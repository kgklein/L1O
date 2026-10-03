"""Retrieve spacecraft ephemerides from NASA SSCWeb."""

from __future__ import annotations

from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from sscws.coordinates import CoordinateSystem
from sscws.sscws import SscWs


TimeLike = Union[str, datetime, pd.Timestamp]


class EphemerisFetchError(RuntimeError):
    """Raised when SSCWeb cannot provide a usable ephemeris."""


def _as_utc_timestamp(value: TimeLike) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _coordinate_system(name: str) -> CoordinateSystem:
    try:
        return CoordinateSystem[name.upper()]
    except KeyError as exc:
        supported = ", ".join(item.name for item in CoordinateSystem)
        raise ValueError(
            f"Unsupported coordinate system {name!r}; expected one of: {supported}."
        ) from exc


def fetch_ephemeris(
    spacecraft_id: str,
    start: TimeLike,
    stop: TimeLike,
    coordinate_system: str = "GSE",
) -> pd.DataFrame:
    """Fetch an SSCWeb ephemeris as UTC timestamps and XYZ positions in km.

    Naive input times are interpreted as UTC. The returned DataFrame has a UTC
    DatetimeIndex named time and x_km, y_km, z_km columns.
    """
    start_time = _as_utc_timestamp(start)
    stop_time = _as_utc_timestamp(stop)
    if start_time >= stop_time:
        raise ValueError("Ephemeris start time must be before stop time.")

    frame = _coordinate_system(coordinate_system)
    time_range = [start_time.isoformat(), stop_time.isoformat()]

    try:
        result = SscWs().get_locations([spacecraft_id], time_range, [frame])
    except Exception as exc:
        raise EphemerisFetchError(
            f"SSCWeb request failed for {spacecraft_id!r}: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise EphemerisFetchError(
            f"SSCWeb returned an invalid response for {spacecraft_id!r}."
        )
    if result.get("HttpStatus") != 200:
        detail = (
            result.get("ErrorDescription")
            or result.get("ErrorMessage")
            or result.get("HttpText")
            or "unknown SSCWeb error"
        )
        raise EphemerisFetchError(
            f"SSCWeb request for {spacecraft_id!r} failed "
            f"(HTTP {result.get('HttpStatus')}): {detail}"
        )

    data_items = result.get("Data") or []
    data = next(
        (item for item in data_items if item.get("Id") == spacecraft_id),
        data_items[0] if data_items else None,
    )
    if data is None:
        raise EphemerisFetchError(
            f"SSCWeb returned no ephemeris for {spacecraft_id!r}."
        )

    coordinates = data.get("Coordinates") or []
    coords = next(
        (
            item
            for item in coordinates
            if getattr(item.get("CoordinateSystem"), "name", None) == frame.name
            or str(getattr(item.get("CoordinateSystem"), "value", "")).upper()
            == frame.value.upper()
        ),
        None,
    )
    if coords is None:
        raise EphemerisFetchError(
            f"SSCWeb returned no {frame.name} coordinates for {spacecraft_id!r}."
        )

    times = data.get("Time")
    x_values = coords.get("X")
    y_values = coords.get("Y")
    z_values = coords.get("Z")
    if any(value is None for value in (times, x_values, y_values, z_values)):
        raise EphemerisFetchError(
            f"SSCWeb returned incomplete coordinates for {spacecraft_id!r}."
        )

    lengths = {len(times), len(x_values), len(y_values), len(z_values)}
    if len(lengths) != 1:
        raise EphemerisFetchError(
            f"SSCWeb returned inconsistent coordinate lengths for {spacecraft_id!r}."
        )
    if len(times) == 0:
        raise EphemerisFetchError(
            f"SSCWeb returned no ephemeris samples for {spacecraft_id!r}."
        )

    try:
        index = pd.to_datetime(times, utc=True)
        values = np.column_stack((x_values, y_values, z_values)).astype(float)
    except (TypeError, ValueError) as exc:
        raise EphemerisFetchError(
            f"SSCWeb returned malformed coordinates for {spacecraft_id!r}."
        ) from exc

    ephemeris = pd.DataFrame(
        values,
        index=pd.DatetimeIndex(index, name="time"),
        columns=["x_km", "y_km", "z_km"],
    )
    ephemeris = ephemeris.replace([np.inf, -np.inf], np.nan).dropna()
    ephemeris = ephemeris.sort_index()
    ephemeris = ephemeris[~ephemeris.index.duplicated(keep="first")]
    if ephemeris.empty:
        raise EphemerisFetchError(
            f"SSCWeb returned no usable ephemeris samples for {spacecraft_id!r}."
        )
    return ephemeris
