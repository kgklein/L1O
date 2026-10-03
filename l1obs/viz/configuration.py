"""Plot common-time spacecraft configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from l1obs.config import SPACECRAFT, SPACECRAFT_COLORS
from l1obs.proc.ephemeris import SpacecraftConfiguration


def calculate_centroid(positions: pd.DataFrame) -> pd.Series:
    """Return the arithmetic mean of available XYZ positions in kilometres."""
    columns = ["x_km", "y_km", "z_km"]
    missing = [column for column in columns if column not in positions.columns]
    if missing:
        raise ValueError(f"Positions table is missing columns: {', '.join(missing)}.")
    if positions.empty:
        raise ValueError("Cannot calculate a centroid without spacecraft positions.")

    values = positions[columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Spacecraft positions must contain only finite values.")
    return values.mean(axis=0)


def plot_configuration(
    configuration: SpacecraftConfiguration,
    center: str = "centroid",
    output: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot three projections of a common-time spacecraft configuration."""
    if center != "centroid":
        raise ValueError("Only center='centroid' is currently supported.")

    centroid = calculate_centroid(configuration.positions)
    coordinates = ["x_km", "y_km", "z_km"]
    relative = (
        configuration.positions[coordinates].astype(float)
        - centroid[coordinates]
    ) / 1000.0

    max_extent = float(np.abs(relative.to_numpy()).max())
    limit = max(1.0, max_extent * 1.1)

    projections = [
        ("x_km", "y_km", "X-Y"),
        ("x_km", "z_km", "X-Z"),
        ("y_km", "z_km", "Y-Z"),
    ]
    axis_names = {"x_km": "X", "y_km": "Y", "z_km": "Z"}
    color_by_label = {
        spec.label: SPACECRAFT_COLORS.get(key, "black")
        for key, spec in SPACECRAFT.items()
    }

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for axis, (horizontal, vertical, title) in zip(axes, projections):
        for row_index, row in configuration.positions.iterrows():
            x_value = relative.loc[row_index, horizontal]
            y_value = relative.loc[row_index, vertical]
            label = str(row["spacecraft"])
            axis.scatter(
                x_value,
                y_value,
                color=color_by_label.get(label, "black"),
                s=42,
                zorder=3,
            )
            axis.annotate(
                label,
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        axis.scatter(0.0, 0.0, marker="+", color="black", s=100, zorder=4)
        axis.annotate(
            "Centroid",
            (0.0, 0.0),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=8,
        )
        axis.set_title(f"{configuration.frame} {title}")
        axis.set_xlabel(
            f"{configuration.frame} {axis_names[horizontal]} ($10^3$ km)"
        )
        axis.set_ylabel(
            f"{configuration.frame} {axis_names[vertical]} ($10^3$ km)"
        )
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.3)

    epoch = pd.Timestamp(configuration.time)
    if epoch.tz is None:
        epoch = epoch.tz_localize("UTC")
    else:
        epoch = epoch.tz_convert("UTC")
    fig.suptitle(
        f"Spacecraft configuration at {epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    fig.tight_layout()

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    return fig
