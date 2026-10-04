"""Compare loaded magnetic measurements and instantaneous constellation geometry."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like, to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from l1obs.config import SPACECRAFT, SPACECRAFT_COLORS


def _utc(value) -> pd.Timestamp:
    time = pd.Timestamp(value)
    if pd.isna(time):
        raise ValueError("Times must not contain NaT.")
    return time.tz_localize("UTC") if time.tz is None else time.tz_convert("UTC")


def _prepare(frame, columns, label):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{label} requires a DataFrame with a DatetimeIndex.")
    if frame.empty or frame.index.hasnans or frame.index.has_duplicates:
        raise ValueError(f"{label} requires nonempty, unique timestamps without NaT.")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(missing)}")
    try:
        result = frame.loc[:, list(columns)].astype(float).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} columns must be numeric.") from exc
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index, utc=True), name="time")
    return result.sort_index().replace([np.inf, -np.inf], np.nan)


def _metadata(frame, columns, units, label, explicit_frame):
    frames = [str(frame.attrs[key]).strip().upper()
              for key in ("coordinate_system", "frame") if key in frame.attrs]
    if any(column.lower().endswith("_gse") for column in columns):
        frames.append("GSE")
    if explicit_frame:
        frames.append(explicit_frame)
    if not explicit_frame and not any(key in frame.attrs for key in ("coordinate_system", "frame")):
        raise ValueError(f"{label} needs coordinate metadata or an explicit coordinate_system.")
    if not frames or any(not item for item in frames) or len(set(frames)) != 1:
        raise ValueError(f"{label} coordinate-system mismatch: {frames}")
    unit_metadata = frame.attrs.get("units", units)
    if isinstance(unit_metadata, Mapping):
        unit_metadata = [unit_metadata.get(column, units) for column in columns]
    else:
        unit_metadata = [unit_metadata]
    if any(str(item).strip().casefold() != units.strip().casefold() for item in unit_metadata):
        raise ValueError(f"{label} inconsistent units; expected {units}.")
    if any(column.lower().endswith("_km") for column in columns) and units.strip().casefold() != "km":
        raise ValueError(f"{label} kilometre-suffixed columns require km units.")
    return frames[0]


def _gap_limit(index, override):
    if override is not None:
        limit = pd.Timedelta(override)
        if pd.isna(limit) or limit <= pd.Timedelta(0):
            raise ValueError("Maximum gap durations must be positive.")
        return limit
    if len(index) < 2:
        return None
    return pd.Timedelta(int(3 * np.median(np.diff(index.as_unit("ns").asi8))), unit="ns")


def _gap_breaks(index, limit):
    if limit is None or len(index) < 2:
        return pd.DatetimeIndex([], tz="UTC")
    times = index.as_unit("ns").asi8
    intervals = np.diff(times)
    large = intervals > limit.value
    return pd.to_datetime(times[:-1][large] + intervals[large] // 2, utc=True)


def _align_geometry(positions, start, end, limits):
    """Return common times, six relative XYZ tracks, and their instantaneous center."""
    valid_indices = [frame.index[np.isfinite(frame.to_numpy()).all(axis=1)] for frame in positions]
    if any(len(index) == 0 for index in valid_indices):
        raise ValueError("No simultaneously valid geometry: a spacecraft has no valid positions.")
    lower = max(start, *(index[0] for index in valid_indices))
    upper = min(end, *(index[-1] for index in valid_indices))
    if lower > upper:
        raise ValueError("Non-overlapping position time coverage in the displayed interval.")
    grid = pd.DatetimeIndex([lower, upper])
    for frame, limit in zip(positions, limits):
        grid = grid.union(frame.index).union(_gap_breaks(frame.index, limit))
    grid = grid[(grid >= lower) & (grid <= upper)].sort_values().unique()
    origin = grid[0].value
    target = (grid.as_unit("ns").asi8 - origin).astype(float)
    aligned = np.full((6, len(grid), 3), np.nan)
    for spacecraft, (frame, limit) in enumerate(zip(positions, limits)):
        values = frame.to_numpy()
        times = frame.index.as_unit("ns").asi8
        valid = np.isfinite(values).all(axis=1)
        # A run ends at every invalid XYZ record or oversized timestamp interval.
        run_start = None
        for row in range(len(frame) + 1):
            boundary = row == len(frame) or not valid[row]
            if row < len(frame) and row > 0 and limit is not None:
                boundary |= times[row] - times[row - 1] > limit.value
            if boundary and run_start is not None:
                run_times = times[run_start:row]
                inside = (grid.as_unit("ns").asi8 >= run_times[0]) & (grid.as_unit("ns").asi8 <= run_times[-1])
                for component in range(3):
                    aligned[spacecraft, inside, component] = np.interp(
                        target[inside], (run_times - origin).astype(float), values[run_start:row, component]
                    )
                run_start = None
            if row < len(frame) and valid[row] and run_start is None:
                run_start = row
    complete = np.isfinite(aligned).all(axis=(0, 2))
    if not complete.any():
        raise ValueError("No simultaneously valid geometry samples for all six spacecraft.")
    aligned[:, ~complete, :] = np.nan
    center = aligned.mean(axis=0)
    relative = aligned - center[None, :, :]
    return grid, relative, center


def _spacecraft_colors(names, overrides):
    known = {key.casefold(): color for key, color in SPACECRAFT_COLORS.items()}
    for key, spec in SPACECRAFT.items():
        known[spec.label.casefold()] = SPACECRAFT_COLORS[key]
    result = {name: known[name.casefold()] for name in names if name.casefold() in known}
    used = {to_hex(color) for color in result.values()}
    palette = [to_hex(color) for color in plt.get_cmap("tab20").colors if to_hex(color) not in used]
    for name, color in zip(sorted(set(names) - result.keys()), palette):
        result[name] = color
    for name in names:
        if overrides and name in overrides:
            result[name] = overrides[name]
        if not is_color_like(result[name]):
            raise ValueError(f"Invalid spacecraft color for {name}: {result[name]}")
    return result


def plot_six_spacecraft_B_and_geometry(
    datasets,
    spacecraft_names=None,
    time_range=None,
    *,
    B_components=("bx_gse", "by_gse", "bz_gse"),
    position_components=("x_km", "y_km", "z_km"),
    B_units="nT",
    position_units="km",
    coordinate_system=None,
    common_B_ylim=True,
    magnitude_column=None,
    magnetic_max_gap=None,
    position_max_gap=None,
    colors=None,
    output_path=None,
    save_pdf=False,
    show=False,
) -> Figure:
    """Plot six native magnetic series and three relative geometry projections.

    Each datasets[name] contains magnetic and positions DataFrames. The center
    is the instantaneous arithmetic mean of all six valid XYZ vectors; any
    incomplete geometry sample is masked for every spacecraft. Positions are
    linearly interpolated on their timestamp union within shared coverage, never
    across NaNs or oversized gaps and never extrapolated. Gap limits default to
    three times each input's median sample spacing; timedelta-like overrides
    apply separately to magnetic and position data. Magnetic data are not
    interpolated. All vectors must share the declared/metadata coordinate frame;
    missing metadata requires coordinate_system, and no frame/unit conversion
    occurs. Magnitude defaults to the displayed vector norm; magnitude_column
    explicitly selects an archive scalar instead. Return an open Figure, saving
    a 300-dpi PNG and optionally a sibling PDF when output_path is supplied.
    """
    if not isinstance(datasets, Mapping) or len(datasets) != 6:
        raise ValueError("Exactly six spacecraft datasets are required.")
    names = list(datasets) if spacecraft_names is None else list(spacecraft_names)
    if len(names) != 6 or len(set(names)) != 6 or set(names) != set(datasets):
        raise ValueError("spacecraft_names must be a permutation of the six dataset names.")
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("Spacecraft names must be nonempty strings.")
    if len(B_components) != 3 or len(set(B_components)) != 3 or len(position_components) != 3 or len(set(position_components)) != 3:
        raise ValueError("Field and position components must each contain three distinct columns.")
    if not isinstance(B_units, str) or not B_units.strip() or not isinstance(position_units, str) or not position_units.strip():
        raise ValueError("Field and position units must be nonempty strings.")
    output = Path(output_path) if output_path is not None else None
    if output is not None and output.suffix.lower() != ".png":
        raise ValueError("output_path must be a PNG path.")
    if save_pdf and output is None:
        raise ValueError("save_pdf requires output_path.")
    explicit_frame = str(coordinate_system).strip().upper() if coordinate_system is not None else None
    if coordinate_system is not None and not explicit_frame:
        raise ValueError("coordinate_system must be nonempty.")
    magnetic, positions, magnetic_limits, position_limits, frames = [], [], [], [], []
    field_columns = list(B_components) + ([magnitude_column] if magnitude_column is not None else [])
    for name in names:
        entry = datasets[name]
        if not isinstance(entry, Mapping) or "magnetic" not in entry or "positions" not in entry:
            raise ValueError(f"{name} requires separate magnetic and positions DataFrames.")
        for kind, columns, units, destination, limits, override in (
            ("magnetic", field_columns, B_units, magnetic, magnetic_limits, magnetic_max_gap),
            ("positions", position_components, position_units, positions, position_limits, position_max_gap),
        ):
            source = entry[kind]
            prepared = _prepare(source, columns, f"{name} {kind}")
            frames.append(_metadata(source, columns, units, f"{name} {kind}", explicit_frame))
            destination.append(prepared)
            limits.append(_gap_limit(prepared.index, override))
    if len(set(frames)) != 1:
        raise ValueError(f"Coordinate-system mismatch between spacecraft/vectors: {frames}")
    frame_name = frames[0]
    if time_range is None:
        start = max(frame.index[0] for frame in magnetic)
        end = min(frame.index[-1] for frame in magnetic)
    else:
        if len(time_range) != 2:
            raise ValueError("time_range must contain start and end.")
        start, end = map(_utc, time_range)
    if start >= end:
        raise ValueError("Displayed time range must be nonempty with start before end.")
    tracks = []
    for name, field, limit in zip(names, magnetic, magnetic_limits):
        clipped = field.loc[start:end]
        if clipped.empty:
            raise ValueError(f"{name} has no magnetic samples in the displayed interval.")
        breaks = _gap_breaks(field.index, limit)
        breaks = breaks[(breaks >= start) & (breaks <= end)]
        clipped = clipped.reindex(clipped.index.union(breaks))
        vector = clipped.loc[:, list(B_components)].to_numpy()
        magnitude = (np.linalg.norm(vector, axis=1) if magnitude_column is None
                     else clipped[magnitude_column].to_numpy())
        tracks.append((clipped.index, np.column_stack([magnitude, vector])))
    geometry_times, relative, _ = _align_geometry(positions, start, end, position_limits)
    spacecraft_colors = _spacecraft_colors(names, colors)
    styles = [("black", "-", "|B|"), ("#1f77b4", "-", "Bx"),
              ("#d62728", "--", "By"), ("#2ca02c", ":", "Bz")]
    with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                         "legend.fontsize": 8, "lines.linewidth": 0.9,
                         "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(14, 11), layout="constrained")
        outer = fig.add_gridspec(1, 2, width_ratios=(2, 1), wspace=0.08)
        left = outer[0].subgridspec(6, 1, hspace=0.04)
        right = outer[1].subgridspec(3, 1, hspace=0.12)
        magnetic_axes = []
        for row, (name, (times, values)) in enumerate(zip(names, tracks)):
            axis = fig.add_subplot(left[row], sharex=magnetic_axes[0] if row else None)
            magnetic_axes.append(axis)
            for component, (color, linestyle, label) in enumerate(styles):
                axis.plot(times, values[:, component], color=color, linestyle=linestyle, label=label)
            axis.text(0.015, 0.9, name, transform=axis.transAxes, va="top",
                      color=spacecraft_colors[name], fontweight="bold",
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
            if not np.isfinite(values).any():
                axis.text(0.5, 0.5, "No valid B samples", transform=axis.transAxes, ha="center")
            axis.set_ylabel(f"B [{B_units}]")
            axis.grid(True, alpha=0.18)
            axis.tick_params(labelbottom=row == 5)
        magnetic_axes[0].legend(ncol=4, loc="lower right", bbox_to_anchor=(1, 1.02), frameon=False)
        magnitude_label = "vector norm" if magnitude_column is None else magnitude_column
        magnetic_axes[0].set_title(f"{frame_name} components; |B|: {magnitude_label}", loc="left")
        magnetic_axes[-1].set_xlim(start, end)
        locator = mdates.AutoDateLocator(tz=timezone.utc)
        magnetic_axes[-1].xaxis.set_major_locator(locator)
        magnetic_axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=timezone.utc))
        magnetic_axes[-1].set_xlabel("Time [UTC]")
        if common_B_ylim:
            all_values = np.concatenate([values.ravel() for _, values in tracks])
            finite = all_values[np.isfinite(all_values)]
            if finite.size:
                lower, upper = finite.min(), finite.max()
                padding = 0.06 * (upper - lower if upper > lower else max(abs(upper), 1))
                for axis in magnetic_axes:
                    axis.set_ylim(lower - padding, upper + padding)
        extent = np.nanmax(np.abs(relative))
        distance_limit = 1.1 * extent if extent > 0 else 1.0
        valid_times = np.flatnonzero(np.isfinite(relative).all(axis=(0, 2)))
        geometry_axes = []
        for row, (horizontal, vertical) in enumerate(((0, 1), (0, 2), (1, 2))):
            axis = fig.add_subplot(right[row])
            geometry_axes.append(axis)
            for spacecraft, name in enumerate(names):
                color = spacecraft_colors[name]
                track = relative[spacecraft]
                axis.plot(track[:, horizontal], track[:, vertical], color=color, label=name)
                axis.plot(track[valid_times[0], horizontal], track[valid_times[0], vertical],
                          marker="o", markersize=5, markerfacecolor="none", color=color, linestyle="none")
                axis.plot(track[valid_times[-1], horizontal], track[valid_times[-1], vertical],
                          marker="^", markersize=3.5, color=color, linestyle="none")
            axis.axhline(0, color="black", alpha=0.15, linewidth=0.6)
            axis.axvline(0, color="black", alpha=0.15, linewidth=0.6)
            axis.plot(0, 0, marker="+", color="black", markersize=7, linestyle="none")
            axis.set(xlabel=f"Δ{'xyz'[horizontal]} [{position_units}]",
                     ylabel=f"Δ{'xyz'[vertical]} [{position_units}]",
                     xlim=(-distance_limit, distance_limit), ylim=(-distance_limit, distance_limit))
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, alpha=0.12)
            axis.set_title(f"{frame_name} {'xyz'[horizontal]}–{'xyz'[vertical]}")
        handles = [Line2D([], [], color=spacecraft_colors[name], label=name) for name in names]
        geometry_axes[0].legend(handles=handles, ncol=3, loc="lower center",
                                bbox_to_anchor=(0.5, 1.1), frameon=False)
        first, last = geometry_times[valid_times[0]], geometry_times[valid_times[-1]]
        geometry_axes[-1].text(
            0.5, -0.27,
            "○ Start   ▲ End   + Mesocenter\n"
            f"{first.strftime('%Y-%m-%d %H:%M:%S.%f').rstrip('0').rstrip('.')} →\n"
            f"{last.strftime('%Y-%m-%d %H:%M:%S.%f').rstrip('0').rstrip('.')} UTC",
            transform=geometry_axes[-1].transAxes, ha="center", va="top", fontsize=8,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=300, bbox_inches="tight")
            if save_pdf:
                fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        if show:
            plt.show()
    return fig
