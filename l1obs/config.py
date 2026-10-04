from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpacecraftSpec:
    """Project-wide metadata for a spacecraft."""

    label: str
    ssc_id: str


# Canonical spacecraft registry. Dictionary order is the default display order.
SPACECRAFT = {
    "WIND": SpacecraftSpec(label="Wind", ssc_id="wind"),
    "ACE": SpacecraftSpec(label="ACE", ssc_id="ace"),
    "DSCOVR": SpacecraftSpec(label="DSCOVR", ssc_id="dscovr"),
    "ADITYA-L1": SpacecraftSpec(label="Aditya-L1", ssc_id="adityal1"),
    "IMAP": SpacecraftSpec(label="IMAP", ssc_id="imap"),
    "SOLAR-1": SpacecraftSpec(label="SOLAR-1", ssc_id="solar1"),
}

# --- Stable, project-wide spacecraft colors (edit once, then never change) ---
SPACECRAFT_COLORS = {
    "WIND":   "#1f77b4",  # blue
    "ACE":    "#d62728",  # red
    "DSCOVR": "#2ca02c",  # green
    "SOHO":   "#ff7f0e",  # orange
    "IMAP":   "#0a0a0a",  # black
    "SWFO":   "#17becf",  # teal (reserved)
    "ADITYA": "#9467bd",  # legacy key
    "ADITYA-L1": "#9467bd",  # purple
    "SOLAR-1": "#8c564b",  # brown
}

# --- Science products and optional provider (CDAWeb by default) ---
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
    # IMAP (magnetic field only)
    "IMAP": {
        "mag": "IMAP_MAG_L2_NORM-GSE",  # 0.5-sec
    },
    "SOLAR-1": {
        "mag": "sci_mag-l3_solar1",  # science-quality, 1-sec
        "mag_provider": "ncei",
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
