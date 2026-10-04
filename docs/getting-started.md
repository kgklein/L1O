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

Retrieval examples need network access to NASA SSCWeb or CDAWeb and a date covered
by the requested archive. CDAWeb requests reuse a local parquet cache unless
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

Use the dataset-level interface; there is no separate mission-name magnetic
wrapper. This example fetches Wind's native 3-second product:

```python
from pathlib import Path
import pandas as pd
from l1obs.fetch.cdaweb import fetch_cdaweb_dataset

result = fetch_cdaweb_dataset(
    "WI_H0_MFI",
    pd.Timestamp("2026-09-01T00:00:00Z"),
    pd.Timestamp("2026-09-01T00:10:00Z"),
    cache_dir=Path("cache"),
)
magnetic = result.df
print(magnetic.head())
```

For ACE use `AC_H3_MFI`; for DSCOVR use `DSCOVR_H0_MAG`. Both preserve native
1-second samples. The returned columns are `bx_gse`, `by_gse`, `bz_gse`, and
`b_mag`, in nT, on a UTC index named `time`. Invalid samples remain as NaNs;
fetching does not resample. See [Data Products](data-products.md).

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

## View these docs locally

Documentation tools are separate from the package dependencies. From the
repository root:

```bash
python -m pip install mkdocs-material
mkdocs serve
```

Open the local address printed by MkDocs. To generate the static site instead,
run `mkdocs build`; its default output directory is `site/`.
