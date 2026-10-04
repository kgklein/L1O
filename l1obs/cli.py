from __future__ import annotations

import argparse
import json
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
from l1obs.proc.coords import gse_to_pseudo_rtn, ensure_v_rtn
from l1obs.proc.ephemeris import get_positions_at_time
from l1obs.proc.merge import merge_frames
from l1obs.proc.resample import resample_to_1s
from l1obs.viz.configuration import calculate_centroid, plot_configuration
from l1obs.viz.constellation import plot_six_spacecraft_B_and_geometry
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


def _magnetic_columns(df: pd.DataFrame, spacecraft: str) -> pd.DataFrame:
    """Prefix normalized GSE data and retain the legacy pseudo-RTN interface."""
    out = df.add_prefix(f"{spacecraft}_")
    rtn = gse_to_pseudo_rtn(df[["bx_gse", "by_gse", "bz_gse"]].to_numpy())
    for component, values in zip(["Br", "Bt", "Bn"], rtn.T):
        out[f"{spacecraft}_{component}"] = values
    out[f"{spacecraft}_Bmag"] = df["b_mag"]
    return out


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
            magnetic = _magnetic_columns(result.df, spacecraft)
            frames.append(resample_to_1s(
                magnetic, t0, t1, preserve_nan_gaps=True
            ))

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
            frames.append(resample_to_1s(plasma.sort_index(), t0, t1))

        if not frames:
            continue

        spacecraft_frame = frames[0]
        for extra in frames[1:]:
            spacecraft_frame = spacecraft_frame.join(extra, how="outer")
        frames_1s[spacecraft] = spacecraft_frame.sort_index()

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


def _positive_gap(value: str) -> pd.Timedelta:
    try:
        gap = pd.Timedelta(value)
        if pd.isna(gap) or gap <= pd.Timedelta(0):
            raise ValueError("duration must be positive")
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Invalid gap duration {value!r}: {exc}") from exc
    return gap


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key!r}")
        result[key] = value
    return result


def _load_constellation_manifest(manifest_path: Path) -> dict:
    """Load six pairs of local tables, resolving paths relative to the manifest."""
    try:
        manifest = json.loads(manifest_path.read_text(), object_pairs_hook=_unique_json_object)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read constellation manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or len(manifest) != 6:
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


def _run_constellation(args: argparse.Namespace) -> int:
    if (args.start is None) != (args.end is None):
        raise RuntimeError("Supply --start and --end together, or omit both.")
    datasets = _load_constellation_manifest(args.manifest)
    output = args.output if args.output is not None else default_paths().output / "constellation.png"
    try:
        figure = plot_six_spacecraft_B_and_geometry(
            datasets,
            spacecraft_names=args.spacecraft_order,
            time_range=(args.start, args.end) if args.start is not None else None,
            B_components=tuple(args.b_components),
            position_components=tuple(args.position_components),
            B_units=args.b_units,
            position_units=args.position_units,
            coordinate_system=args.coordinate_system,
            common_B_ylim=not args.independent_b_ylim,
            magnitude_column=args.magnitude_column,
            magnetic_max_gap=args.magnetic_max_gap,
            position_max_gap=args.position_max_gap,
            output_path=output,
            save_pdf=args.pdf,
            show=args.show,
        )
    except (ValueError, TypeError, OSError) as exc:
        raise RuntimeError(f"Cannot plot constellation: {exc}") from exc
    plt.close(figure)
    print(f"Wrote: {output}")
    if args.pdf:
        print(f"Wrote: {output.with_suffix('.pdf')}")
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

    constellation = subparsers.add_parser(
        "constellation", help="Plot six spacecraft from a local JSON manifest; no downloads."
    )
    constellation.add_argument("--manifest", type=Path, required=True,
                               help="JSON mapping six names to magnetic/positions parquet or CSV paths.")
    constellation.add_argument("--output", type=Path,
                               help="PNG output (default ./output/constellation.png).")
    constellation.add_argument("--pdf", action="store_true", help="Also save a sibling PDF.")
    constellation.add_argument("--show", action="store_true", help="Display the figure after saving.")
    constellation.add_argument("--start", type=_parse_utc_time, help="UTC range start; requires --end.")
    constellation.add_argument("--end", type=_parse_utc_time, help="UTC range end; requires --start.")
    constellation.add_argument("--coordinate-system", help="Common frame; required without file metadata.")
    constellation.add_argument("--spacecraft-order", nargs=6, metavar="NAME",
                               help="Display order; must list the six manifest names exactly once.")
    constellation.add_argument("--b-components", nargs=3, default=("bx_gse", "by_gse", "bz_gse"),
                               metavar=("BX", "BY", "BZ"), help="Magnetic component column names.")
    constellation.add_argument("--position-components", nargs=3, default=("x_km", "y_km", "z_km"),
                               metavar=("X", "Y", "Z"), help="Position component column names.")
    constellation.add_argument("--b-units", default="nT", help="Magnetic input units (default nT).")
    constellation.add_argument("--position-units", default="km", help="Position input units (default km).")
    constellation.add_argument("--independent-b-ylim", action="store_true",
                               help="Autoscale magnetic panels independently.")
    constellation.add_argument("--magnitude-column", help="Use this archive column instead of the vector norm.")
    constellation.add_argument("--magnetic-max-gap", type=_positive_gap, help="Magnetic gap limit, e.g. 5s.")
    constellation.add_argument("--position-max-gap", type=_positive_gap, help="Position gap limit, e.g. 5min.")
    constellation.set_defaults(handler=_run_constellation)

    for subparser in (timeseries, positions, constellation):
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
