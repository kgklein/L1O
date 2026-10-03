from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from l1obs.config import DATASETS, SPACECRAFT_COLORS, default_paths
from l1obs.fetch.cdaweb import fetch_cdaweb_dataset
from l1obs.io.save import save_hdf5
from l1obs.logging_utils import setup_logging
from l1obs.proc.coords import ensure_b_rtn, ensure_v_rtn
from l1obs.proc.ephemeris import get_positions_at_time
from l1obs.proc.merge import merge_frames
from l1obs.proc.resample import resample_to_1s
from l1obs.viz.configuration import calculate_centroid, plot_configuration
from l1obs.viz.timeseries import plot_timeseries


def _parse_utc_time(value: str) -> pd.Timestamp:
    """Parse an ISO-like time, interpreting a naive value as UTC."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Invalid time {value!r}: {exc}") from exc
    if pd.isna(timestamp):
        raise argparse.ArgumentTypeError(f"Invalid time {value!r}.")
    if timestamp.tz is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _run_timeseries(args: argparse.Namespace) -> int:
    log = logging.getLogger("l1obs.cli")
    paths = default_paths()
    outdir = Path(args.outdir) if args.outdir else paths.output
    cachedir = Path(args.cachedir) if args.cachedir else paths.cache
    outdir.mkdir(parents=True, exist_ok=True)
    cachedir.mkdir(parents=True, exist_ok=True)

    t0 = args.start.floor("h")
    t1 = t0 + pd.Timedelta(hours=1)
    log.info("Hour window: %s to %s", t0.isoformat(), t1.isoformat())
    log.info("Cache dir: %s", cachedir)
    log.info("Output dir: %s", outdir)

    frames_1s: Dict[str, pd.DataFrame] = {}
    for spacecraft, spec in DATASETS.items():
        frames = []
        log.info("Processing spacecraft: %s", spacecraft)

        if "mag" in spec:
            result = fetch_cdaweb_dataset(
                spec["mag"], t0, t1, cachedir, force=args.force
            )
            magnetic = result.df.copy()
            for column in magnetic.columns:
                if magnetic[column].dtype == "object":
                    magnetic = ensure_b_rtn(magnetic, column, spacecraft)
                    break
            frames.append(magnetic)

        if "plasma" in spec:
            result = fetch_cdaweb_dataset(
                spec["plasma"], t0, t1, cachedir, force=args.force
            )
            plasma = result.df.copy()
            for column in plasma.columns:
                if plasma[column].dtype == "object":
                    plasma = ensure_v_rtn(plasma, column, spacecraft)
                    break
            for candidate in ["Np", "N_p", "DENSITY", "proton_density", "N_P"]:
                if candidate in plasma.columns:
                    plasma[f"{spacecraft}_Np"] = pd.to_numeric(
                        plasma[candidate], errors="coerce"
                    )
                    break
            for candidate in ["Tp", "T_p", "TEMP", "proton_temperature", "T_P"]:
                if candidate in plasma.columns:
                    plasma[f"{spacecraft}_Tp"] = pd.to_numeric(
                        plasma[candidate], errors="coerce"
                    )
                    break
            frames.append(plasma)

        if not frames:
            continue

        spacecraft_frame = frames[0]
        for extra in frames[1:]:
            spacecraft_frame = spacecraft_frame.join(extra, how="outer")
        frames_1s[spacecraft] = resample_to_1s(
            spacecraft_frame.sort_index(), t0, t1
        )

    merged = merge_frames(frames_1s)
    stamp = t0.strftime("%Y%m%d_%HUT")
    h5path = outdir / f"L1_Observatory_{stamp}.h5"
    plotpath = outdir / f"L1_Observatory_{stamp}_timeseries.png"

    log.info("Merging %d spacecraft frames", len(frames_1s))
    log.info("Saving merged dataset: %s", h5path)
    save_hdf5(merged, h5path)
    log.info("Saving plot: %s", plotpath)
    plot_timeseries(
        merged,
        spacecraft=list(frames_1s.keys()),
        colors=SPACECRAFT_COLORS,
        outpath=plotpath,
        title=f"L1 Observatory | {t0.isoformat()} to {t1.isoformat()}",
    )

    print(f"Wrote: {h5path}")
    print(f"Wrote: {plotpath}")
    return 0


def _run_positions(args: argparse.Namespace) -> int:
    configuration = get_positions_at_time(args.time)
    centroid = calculate_centroid(configuration.positions)
    table = configuration.positions[["spacecraft", "x_km", "y_km", "z_km"]].copy()
    table["dx_km"] = table["x_km"] - centroid["x_km"]
    table["dy_km"] = table["y_km"] - centroid["y_km"]
    table["dz_km"] = table["z_km"] - centroid["z_km"]

    print(
        f"{configuration.frame} positions at "
        f"{pd.Timestamp(configuration.time).isoformat()}"
    )
    print(table.to_string(index=False, float_format=lambda value: f"{value:,.1f}"))

    for name, message in configuration.errors.items():
        print(f"Warning: {name}: {message}", file=sys.stderr)

    figure = plot_configuration(configuration, output=args.output)
    if args.output is None:
        plt.show()
    else:
        print(f"Wrote: {args.output}")
    plt.close(figure)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="L1 Observatory data and spacecraft configuration tools."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    timeseries = subparsers.add_parser(
        "timeseries",
        help="Fetch, merge, and plot an hour of solar-wind observations.",
    )
    timeseries.add_argument(
        "--start",
        required=True,
        type=_parse_utc_time,
        help="UTC start time; will be floored to the hour.",
    )
    timeseries.add_argument(
        "--outdir", default=None, help="Output directory (default ./output)."
    )
    timeseries.add_argument(
        "--cachedir", default=None, help="Cache directory (default ./cache)."
    )
    timeseries.add_argument(
        "--force", action="store_true", help="Force re-download even if cached."
    )
    timeseries.set_defaults(handler=_run_timeseries)

    positions = subparsers.add_parser(
        "positions",
        help="Fetch and plot common-time spacecraft positions.",
    )
    positions.add_argument(
        "--time",
        required=True,
        type=_parse_utc_time,
        help="Requested UTC epoch in ISO format.",
    )
    positions.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path; display interactively when omitted.",
    )
    positions.set_defaults(handler=_run_positions)

    for subparser in (timeseries, positions):
        subparser.add_argument(
            "-v", "--verbose", action="store_true", help="Print progress."
        )
        subparser.add_argument(
            "-vv",
            action="store_true",
            help="Enable debug-level progress.",
        )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(verbose=args.verbose, very_verbose=args.vv)
    logging.getLogger("l1obs.cli").info("Starting l1obs %s", args.command)

    try:
        status = args.handler(args)
    except RuntimeError as exc:
        parser.exit(1, f"Error: {exc}\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
