from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from l1obs.config import DATASETS, SPACECRAFT_COLORS, default_paths
from l1obs.fetch.cdaweb import fetch_cdaweb_dataset
from l1obs.proc.coords import ensure_b_rtn, ensure_v_rtn
from l1obs.proc.resample import resample_to_1s
from l1obs.proc.merge import merge_frames
from l1obs.io.save import save_hdf5
from l1obs.viz.timeseries import plot_timeseries


def _parse_time(s: str) -> pd.Timestamp:
    # Expect ISO-like input; treat as UTC
    t = pd.Timestamp(s)
    if t.tz is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    # snap to hour start
    return t.floor("h")


def main() -> None:
    p = argparse.ArgumentParser(description="L1 Observatory: fetch + merge + plot for a selected hour.")
    p.add_argument("--start", required=True, help="UTC start time (any ISO format); will be floored to the hour.")
    p.add_argument("--outdir", default=None, help="Output directory (default ./output)")
    p.add_argument("--cachedir", default=None, help="Cache directory (default ./cache)")
    p.add_argument("--force", action="store_true", help="Force re-download even if cached.")
    args = p.parse_args()

    paths = default_paths()
    outdir = Path(args.outdir) if args.outdir else paths.output
    cachedir = Path(args.cachedir) if args.cachedir else paths.cache
    outdir.mkdir(parents=True, exist_ok=True)
    cachedir.mkdir(parents=True, exist_ok=True)

    t0 = _parse_time(args.start)
    t1 = t0 + pd.Timedelta(hours=1)

    frames_1s: Dict[str, pd.DataFrame] = {}

    for sc, spec in DATASETS.items():
        # Fetch MAG (if available) and PLASMA (if available), then assemble a per-spacecraft DF
        dfs = []

        if "mag" in spec:
            fr = fetch_cdaweb_dataset(spec["mag"], t0, t1, cachedir, force=args.force)
            dfmag = fr.df.copy()
            # Try to standardize B vector: prefer RTN var, else GSE var (pseudo RTN)
            # We don't know exact column names a priori, so try both logicals if present.
            # If dataset returned a column that is the vector, it will be that column name.
            # We’ll pick the first vector-like column that exists.
            for col in dfmag.columns:
                # vector columns often store array-like objects; crude check
                if dfmag[col].dtype == "object":
                    # treat as B vector candidate
                    dfmag = ensure_b_rtn(dfmag, col, sc)
                    break
            dfs.append(dfmag)

        if "plasma" in spec:
            fr = fetch_cdaweb_dataset(spec["plasma"], t0, t1, cachedir, force=args.force)
            dfpl = fr.df.copy()
            # Try velocity vector standardization
            for col in dfpl.columns:
                if dfpl[col].dtype == "object":
                    dfpl = ensure_v_rtn(dfpl, col, sc)
                    break
            # Try density + temperature standardization (best effort: rename if exact names present)
            # If the dataset already uses common names (Np/Tp), just prefix them.
            for cand in ["Np", "N_p", "DENSITY", "proton_density", "N_P"]:
                if cand in dfpl.columns:
                    dfpl[f"{sc}_Np"] = pd.to_numeric(dfpl[cand], errors="coerce")
                    break
            for cand in ["Tp", "T_p", "TEMP", "proton_temperature", "T_P"]:
                if cand in dfpl.columns:
                    dfpl[f"{sc}_Tp"] = pd.to_numeric(dfpl[cand], errors="coerce")
                    break

            dfs.append(dfpl)

        if not dfs:
            continue

        # Merge mag+plasma for this spacecraft on time
        df_sc = dfs[0]
        for extra in dfs[1:]:
            df_sc = df_sc.join(extra, how="outer")

        df_sc = df_sc.sort_index()
        # Resample to 1-second cadence
        df_sc_1s = resample_to_1s(df_sc, t0, t1)

        # Keep only standardized columns + maybe originals later (we keep originals for now; you can prune later)
        frames_1s[sc] = df_sc_1s

    merged = merge_frames(frames_1s)

    # Save merged HDF5
    stamp = t0.strftime("%Y%m%d_%HUT")
    h5path = outdir / f"L1_Observatory_{stamp}.h5"
    save_hdf5(merged, h5path)

    # Plot
    plotpath = outdir / f"L1_Observatory_{stamp}_timeseries.png"
    plot_timeseries(
        merged,
        spacecraft=list(frames_1s.keys()),
        colors=SPACECRAFT_COLORS,
        outpath=plotpath,
        title=f"L1 Observatory | {t0.isoformat()} to {t1.isoformat()}",
    )

    print(f"Wrote: {h5path}")
    print(f"Wrote: {plotpath}")
