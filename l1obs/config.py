from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Stable, project-wide spacecraft colors (edit once, then never change) ---
SPACECRAFT_COLORS = {
    "WIND":   "#1f77b4",  # blue
    "ACE":    "#d62728",  # red
    "DSCOVR": "#2ca02c",  # green
    "SOHO":   "#ff7f0e",  # orange
    "IMAP":   "#0a0a0a",  # black (reserved)
    "SWFO":   "#17becf",  # teal (reserved)
    "ADITYA": "#9467bd",  # purple (reserved)
}

# --- CDAWeb dataset IDs (public now) ---
DATASETS = {
    # ACE
    "ACE": {
        "mag":    "AC_H3_MFI",  # 1-sec MAG (includes multiple coordinate frames)
        "plasma": "AC_H0_SWE",  # 64-sec SWEPAM
    },
    # Wind
    "WIND": {
        "mag":    "WI_H0_MFI",  # ~3-sec avg
        "plasma": "WI_PM_3DP",  # ~3-sec proton moments
    },
    # DSCOVR
    "DSCOVR": {
        "mag":    "DSCOVR_H0_MAG",  # 1-sec
        "plasma": "DSCOVR_H1_FC",   # 1-min
    },
    # SOHO (plasma only)
    "SOHO": {
        "plasma": "SOHO_CELIAS-PM_30S",  # 30-sec proton monitor
    },
}

@dataclass(frozen=True)
class Paths:
    root: Path
    cache: Path
    output: Path

def default_paths() -> Paths:
    root = Path.cwd()
    return Paths(
        root=root,
        cache=root / "cache",
        output=root / "output",
    )
