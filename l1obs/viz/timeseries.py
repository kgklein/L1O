from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def _pick_existing(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def plot_timeseries(
    df: pd.DataFrame,
    spacecraft: List[str],
    colors: Dict[str, str],
    outpath: Path,
    title: str,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(12, 10))

    # Panels: Bmag, Vr, Np, Tp (best-effort)
    for sc in spacecraft:
        c = colors.get(sc, "k")

        bcols = _pick_existing(df, [f"{sc}_Bmag"])
        vcols = _pick_existing(df, [f"{sc}_Vr", f"{sc}_Vmag"])
        ncols = _pick_existing(df, [f"{sc}_Np"])
        tcols = _pick_existing(df, [f"{sc}_Tp"])

        if bcols:
            axes[0].plot(df.index, df[bcols[0]], label=sc, color=c)
        if vcols:
            axes[1].plot(df.index, df[vcols[0]], label=sc, color=c)
        if ncols:
            axes[2].plot(df.index, df[ncols[0]], label=sc, color=c)
        if tcols:
            axes[3].plot(df.index, df[tcols[0]], label=sc, color=c)

    axes[0].set_ylabel("|B| (nT)")
    axes[1].set_ylabel("V (km/s)")
    axes[2].set_ylabel("Np (cm$^{-3}$)")
    axes[3].set_ylabel("Tp")
    axes[3].set_xlabel("Time (UTC)")

    axes[0].legend(ncols=4, fontsize=9, loc="upper right")
    axes[0].set_title(title)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
