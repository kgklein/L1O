"""Shock diagnostic figures. No numerical inference is performed here."""
from __future__ import annotations

from datetime import timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from l1obs.config import SPACECRAFT, SPACECRAFT_COLORS
from .data import B_COLUMNS, native_cadence

COMPONENT_STYLES = (("#1f77b4", "-", "Bx"), ("#d62728", "--", "By"), ("#2ca02c", ":", "Bz"))


def save_figure(figure, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(path.with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _broken_profile(frame, columns, max_gap_s=None):
    # Insert plotting-only NaN records; never connect missing timestamp gaps.
    limit = 3 * native_cadence(frame) if max_gap_s is None else max_gap_s
    times = frame.index.as_unit("ns").asi8
    gaps = np.flatnonzero(np.diff(times) / 1e9 > limit)
    extra = pd.to_datetime(times[gaps] + np.diff(times)[gaps] // 2, utc=True)
    return frame[list(columns)].reindex(frame.index.union(extra)).sort_index()


def plot_inspection(magnetic: dict, arrivals: pd.DataFrame | None = None, *, candidates=None,
                    components=False, window=None, relative=False, zoom_s=None,
                    title="Native magnetic measurements", max_gap_s=None):
    """Stack native profiles; relative/individual zoom views alter only the axes."""
    if relative and zoom_s is None:
        zoom_s = 120
    names = list(magnetic)
    figure, axes = plt.subplots(len(names), 1, figsize=(11, 1.7 * len(names) + 1),
                               sharex=(zoom_s is None or relative), squeeze=False,
                               layout="constrained")
    rows = {} if arrivals is None or arrivals.empty else {r["spacecraft"]: r for r in arrivals.to_dict("records")}
    for axis, name in zip(axes[:, 0], names):
        frame = magnetic[name]
        row = rows.get(name)
        center = pd.Timestamp(row["crossing_time"]) if row and pd.notna(row.get("crossing_time")) else None
        profile = _broken_profile(frame, (*B_COLUMNS, "b_mag"), max_gap_s)
        times = (profile.index - center).total_seconds() if relative and center is not None else profile.index
        if relative and center is None:
            axis.text(.5, .5, "No crossing selected", transform=axis.transAxes, ha="center")
        else:
            axis.plot(times, profile.b_mag, color="black", lw=.9, label="|B|")
            if components:
                for column, (color, style, label) in zip(B_COLUMNS, COMPONENT_STYLES):
                    axis.plot(times, profile[column], color=color, ls=style, lw=.7, label=label)
            if center is not None:
                x = 0 if relative else center
                accepted = bool(row.get("accepted", False))
                axis.axvline(x, color="black" if accepted else "#e69f00", ls="-" if accepted else "--",
                             lw=1, label="Accepted" if accepted else "Proposed")
                sigma = row.get("uncertainty_s")
                if sigma is not None and pd.notna(sigma):
                    delta = float(sigma) if relative else pd.Timedelta(seconds=float(sigma))
                    axis.axvspan(x - delta, x + delta, color="grey", alpha=.2, label="±1σ")
            if not relative:
                for candidate in (candidates or {}).get(name, []):
                    axis.axvline(pd.Timestamp(candidate["crossing_time"]), color="#e69f00", ls=":", alpha=.6)
            if relative:
                axis.set_xlim(-zoom_s, zoom_s)
            elif zoom_s is not None and center is not None:
                axis.set_xlim(center - pd.Timedelta(seconds=zoom_s), center + pd.Timedelta(seconds=zoom_s))
            elif window is not None:
                axis.set_xlim(*window)
        axis.text(.012, .94, SPACECRAFT[name].label, color=SPACECRAFT_COLORS[name],
                  transform=axis.transAxes, va="top", weight="bold",
                  bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        axis.set_ylabel("B [nT]")
        axis.grid(alpha=.2)
        if not relative:
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for label, color, style in (("Accepted", "black", "-"), ("Proposed", "#e69f00", "--"),
                                 ("Other candidates", "#e69f00", ":")):
        if label not in labels:
            handles.append(Line2D([], [], color=color, ls=style, label=label))
    axes[0, 0].legend(handles=handles, loc="upper right", ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("t − tᵢ [s] (display only)" if relative else "UTC (date in title)")
    figure.suptitle(title)
    return figure


def plot_geometry(names, positions_km, *, fit=None, title="Crossing-epoch GSE geometry"):
    positions = np.asarray(positions_km)
    centroid = positions.mean(axis=0)
    relative = positions - centroid
    extent = max(float(np.max(np.abs(relative))) * 1.35, 1)
    figure = plt.figure(figsize=(12, 10), layout="constrained")
    ax3 = figure.add_subplot(221, projection="3d")
    projections = [(figure.add_subplot(222), 0, 1), (figure.add_subplot(223), 0, 2),
                   (figure.add_subplot(224), 1, 2)]
    labels = ("Δx GSE [km]", "Δy GSE [km]", "Δz GSE [km]")
    for index, (name, r) in enumerate(zip(names, relative)):
        ax3.scatter(*r, color=SPACECRAFT_COLORS[name], label=SPACECRAFT[name].label)
        for axis, i, j in projections:
            axis.scatter(r[i], r[j], color=SPACECRAFT_COLORS[name])
            axis.annotate(SPACECRAFT[name].label, (r[i], r[j]), xytext=(5, 5 + 9 * (index % 3)), textcoords="offset points", fontsize=8)
    if fit is not None:
        n = np.asarray(fit.normal_gse)
        # At the reference epoch: q·(r-r_ref) = -a. Plot in centroid coordinates.
        plane_center = np.asarray(fit.reference_position_km) - centroid - fit.offset_s * fit.speed_km_s * n
        helper = np.eye(3)[np.argmin(np.abs(n))]
        u = np.cross(n, helper)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        grid = np.linspace(-extent, extent, 2)
        x, y = np.meshgrid(grid, grid)
        surface = plane_center[:, None, None] + u[:, None, None] * x + v[:, None, None] * y
        ax3.plot_surface(*surface, color="grey", alpha=.22)
        ax3.quiver(*plane_center, *n, length=extent * .7, color="black", label="Propagation normal")
        for axis, i, j in projections:
            axis.arrow(plane_center[i], plane_center[j], n[i] * extent * .7, n[j] * extent * .7,
                       width=extent * .006, color="black", length_includes_head=True)
            # Project the plane/box intersection points; the front is at t_ref.
            corners = surface.reshape(3, -1)
            order = [0, 1, 3, 2, 0]
            axis.plot(corners[i, order], corners[j, order], color="grey", alpha=.5)
    ax3.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent),
            xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])
    ticks = np.linspace(-extent, extent, 5)
    ax3.set_xticks(ticks)
    ax3.set_yticks(ticks)
    ax3.set_zticks(ticks)
    ax3.ticklabel_format(style="sci", scilimits=(0, 0), useOffset=False)
    ax3.set_box_aspect((1, 1, 1))
    ax3.legend(fontsize=8)
    for axis, i, j in projections:
        axis.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel=labels[i], ylabel=labels[j])
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.ticklabel_format(style="sci", scilimits=(0, 0), useOffset=False)
        axis.set_aspect("equal", adjustable="box")
        axis.axhline(0, lw=.5, color="grey")
        axis.axvline(0, lw=.5, color="grey")
        axis.grid(alpha=.2)
    figure.suptitle(title + (f"\nFront at {fit.reference_time.isoformat(timespec='milliseconds')}" if fit is not None else ""))
    return figure


def plot_timing_diagnostics(fit, leave_out):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    y = np.arange(len(fit.spacecraft))
    axes[0].errorbar(fit.residuals_s, y, xerr=fit.uncertainties_s, fmt="none", color="grey", capsize=3)
    for i, name in enumerate(fit.spacecraft):
        axes[0].scatter(fit.residuals_s[i], i, color=SPACECRAFT_COLORS[name], zorder=3)
    axes[0].set_yticks(y, [SPACECRAFT[n].label for n in fit.spacecraft])
    axes[0].axvline(0, color="black", lw=.8)
    axes[0].set_xlabel("Observed − predicted arrival [s] (±1σ)")
    axes[0].set_title(f"RMS {fit.rms_residual_s:.3g} s; dof={fit.dof}")
    for axis in axes[1:]:
        axis.set_yticks(np.arange(len(leave_out)), [SPACECRAFT[r["excluded_spacecraft"]].label for r in leave_out])
        axis.set_ylabel("Excluded spacecraft")
    for i, row in enumerate(leave_out):
        color = SPACECRAFT_COLORS[row["excluded_spacecraft"]]
        if row["status"] == "ok":
            axes[1].scatter(row["speed_km_s"], i, color=color)
            axes[2].scatter(row["angular_difference_deg"], i, color=color)
        else:
            for axis in axes[1:]:
                axis.text(.02, i, "failed", transform=axis.get_yaxis_transform(), color="red")
    axes[1].axvline(fit.speed_km_s, color="black", ls="--", label="Full fit")
    axes[1].set_xlabel("Subset speed [km/s]")
    subset_speeds = [r["speed_km_s"] for r in leave_out if r["status"] == "ok"] + [fit.speed_km_s]
    span = max(subset_speeds) - min(subset_speeds)
    minimum_span = max(1., .05 * fit.speed_km_s)
    if span < minimum_span:
        midpoint = (max(subset_speeds) + min(subset_speeds)) / 2
        axes[1].set_xlim(midpoint - minimum_span / 2, midpoint + minimum_span / 2)
    axes[1].ticklabel_format(axis="x", useOffset=False)
    axes[1].legend(fontsize=8)
    axes[2].set_xlabel("Angle from full-fit normal [degrees]")
    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle("Planar timing residuals and leave-one-out sensitivity")
    return figure
