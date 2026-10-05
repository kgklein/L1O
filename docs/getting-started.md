# Getting Started

## Install

Use Python **3.10 or later**. From the repository root, create an environment and
install the package with the dependencies declared in `pyproject.toml`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
l1obs --help
```

Retrieval examples need network access to NASA SSCWeb/CDAWeb or NOAA/NCEI and a date covered
by the requested archive. CDAWeb and NOAA science requests reuse a local parquet cache unless
`force=True` is supplied. Python interfaces are imported from individual modules;
there is no top-level convenience API.

## Spacecraft configuration

Retrieve available positions at one UTC epoch and save the configuration plot:

```bash
l1obs positions --time 2026-09-01T00:00:00Z --output output/configuration.png
```

The command requests all six configured spacecraft, prints positions and offsets
from the centroid, and reports unavailable spacecraft. Without `--output`, it
opens an interactive plot.

In Python, select spacecraft explicitly and inspect partial failures:

```python
from l1obs.proc.ephemeris import get_positions_at_time

configuration = get_positions_at_time(
    "2026-09-01T00:00:00Z", spacecraft=["Wind", "ACE", "DSCOVR"]
)
print(configuration.positions)
print(configuration.errors)
```

The default frame is GSE. Positions are interpolated only when the target is
bracketed by archive samples, or returned directly for an exact timestamp. If no
requested spacecraft succeeds, the call raises `RuntimeError`.

## Native magnetic-field data

Use the shared mission-name interface. This example fetches Wind's native
3-second product:

```python
from pathlib import Path
import pandas as pd
from l1obs.fetch.magnetic import fetch_magnetic_field

result = fetch_magnetic_field(
    "Wind",
    pd.Timestamp("2026-09-01T00:00:00Z"),
    pd.Timestamp("2026-09-01T00:10:00Z"),
    cache_dir=Path("cache"),
)
magnetic = result.df
print(magnetic.head())
```

For ACE or DSCOVR, pass their mission name; both preserve native 1-second
samples. For IMAP, pass `"IMAP"`, which preserves native
0.5-second samples and accepts only `quality_flags == 0` as good science data.
Pass `"SOLAR-1"` to retrieve NOAA/NCEI science-quality `sci_mag-l3_solar1`
with native 1-second timestamps and only `flags_summary == 0` accepted. Its
interval is `[start, end)`; every requested day must have an available file.
No archive guarantees coverage for a particular date. The original dataset-level
`fetch_cdaweb_dataset` remains available for CDAWeb callers.

The returned columns are `bx_gse`, `by_gse`, `bz_gse`, and
`b_mag`, in nT, on a UTC index named `time`. Invalid samples remain as NaNs;
fetching does not resample. See [Data Products](data-products.md).

## Local Aditya-L1 magnetic data

Place already-downloaded ISRO/ISSDC PRADAN Level-2 MAG files directly in the
specified `cache_dir`, named `L2_AL1_MAG_YYYYMMDD_V00.nc`. Then use the same API:

```python
aditya = fetch_magnetic_field(
    "Aditya-L1",
    pd.Timestamp("2026-09-20T00:00:00Z"),
    pd.Timestamp("2026-09-21T00:00:00Z"),
    cache_dir=Path("cache"),
)
print(aditya.df.head())
```

Aditya-L1 reads local files only, preserves native 10-second samples, and derives
`b_mag` from the masked GSE vector. Only `Quality_flag_10s_data == 1` is accepted.
Requests use `[start, end)` and require a local file for every intersecting day.
Automated PRADAN authentication and downloading are not implemented.

## Export and command options

HDF5 export needs PyTables, which is not included in the package's declared
dependencies:

```bash
python -m pip install tables
l1obs positions --help
l1obs timeseries --help
```

See [Input and Output](io.md) for export calls and the current limitations of the
combined magnetic/plasma timeseries command and CDF export.

## Compare six already-loaded spacecraft

With a JSON manifest of six local magnetic/position file pairs, call the
publication-quality comparison plotter directly:

```bash
l1obs constellation --manifest inputs.json --coordinate-system GSE \
  --output output/constellation.png --pdf
```

This command reads parquet or CSV files and makes no downloads. See the
[manifest example and options](io.md#cli-l1obs-constellation) for file layout,
timestamps, and column requirements.

## Prepare a day for the constellation command

From the repository root with L1O installed, place the day's local Aditya-L1
Level-2 MAG file in `cache/` and run:

```bash
python scripts/prepare_constellation.py --date 2026-04-01
```

The script fetches the other five magnetic products and SSCWeb GSE ephemerides.
It reads Aditya-L1 positions from the same local Level-2 file. Magnetic series
retain their native cadence, NaNs, and normalized magnitude. Positions use km;
SSCWeb requests include five minutes of padding for boundary interpolation.

On success, twelve parquet tables and `inputs.json` are written to
`cache/constellation/2026-04-01/`. Plot them with:

```bash
l1obs constellation --manifest cache/constellation/2026-04-01/inputs.json \
  --start 2026-04-01T00:00:00Z --end 2026-04-02T00:00:00Z \
  --coordinate-system GSE --magnitude-column b_mag \
  --output output/constellation_2026-04-01.png --pdf
```

`--magnitude-column b_mag` retains archive magnitudes for the five remote
missions and the derived magnitude for Aditya-L1. Omit it to compute all
magnitudes from the displayed vectors instead.

The script accepts `--cache-dir`, `--output-dir`, and `--force`. Completed tables
are reused on reruns, allowing preparation to resume after an archive failure.
A new manifest is written only after all six pairs succeed. Archive coverage and
SSCWeb mission availability may prevent completion; failures are reported without
substituting another spacecraft or inventing measurements.

## View these docs locally

Documentation tools are separate from the package dependencies. From the
repository root:

```bash
python -m pip install mkdocs-material
mkdocs serve
```

Open the local address printed by MkDocs. To generate the static site instead,
run `mkdocs build`; its default output directory is `site/`.
