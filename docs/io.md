# Input and Output

Use the module-level imports below. Dates shown in retrieval examples are UTC;
archive availability depends on spacecraft and interval. Product layouts are
described in [Data Products](data-products.md).

## SSCWeb ephemeris retrieval

`l1obs.fetch.sscweb.fetch_ephemeris(spacecraft_id, start, stop, coordinate_system="GSE")`
retrieves a position time series. Supply an SSCWeb ID such as `wind`, start/stop
strings, datetimes, or pandas timestamps, and optionally a coordinate frame.
Naive times are treated as UTC. Returns a DataFrame with a UTC `time` index and
`x_km`, `y_km`, `z_km` columns. Unusable responses raise `EphemerisFetchError`.

```python
from l1obs.fetch.sscweb import fetch_ephemeris

ephemeris = fetch_ephemeris(
    "wind", "2026-09-01T00:00:00Z", "2026-09-01T01:00:00Z"
)
print(ephemeris.head())
```

## Common-time configuration

`l1obs.proc.ephemeris.get_positions_at_time(target_time, spacecraft=None, coordinate_system="GSE", window_minutes=30)`
retrieves positions at one target epoch. `spacecraft` accepts configured names
case-insensitively; omitting it requests all six. The fetch window extends
`window_minutes` on each side of the target. Returns a `SpacecraftConfiguration`
containing positions, frame, epoch, and per-spacecraft errors.

```python
from l1obs.proc.ephemeris import get_positions_at_time

configuration = get_positions_at_time(
    "2026-09-01T00:00:00Z", spacecraft=["Wind", "ACE"]
)
print(configuration.positions)
```

## CDAWeb science data and magnetic access

`l1obs.fetch.cdaweb.fetch_cdaweb_dataset(dataset_id, start, end, cache_dir, force=False)`
fetches a configured dataset. Supply UTC pandas timestamps and a `Path` cache
directory. Returns `FetchResult` with `df`, `dataset_id`, and `used_vars`. Requests
reuse parquet caches; `force=True` downloads again. Magnetic caches are versioned
to exclude older, unnormalized products.

Magnetic access uses this same call with `WI_H0_MFI`, `AC_H3_MFI`, or
`DSCOVR_H0_MAG`; fetching preserves native cadence and returns normalized fields.

```python
from pathlib import Path
import pandas as pd
from l1obs.fetch.cdaweb import fetch_cdaweb_dataset

result = fetch_cdaweb_dataset(
    "AC_H3_MFI",
    pd.Timestamp("2026-09-01T00:00:00Z"),
    pd.Timestamp("2026-09-01T00:10:00Z"),
    Path("cache"),
)
print(result.df.head())
```

Configured plasma datasets are `AC_H0_SWE`, `WI_PM_3DP`, `DSCOVR_H1_FC`, and
`SOHO_CELIAS-PM_30S`. Their fetch path expects the legacy pandas-dictionary
response, whereas current CDAWeb clients return a status/data tuple. That path
can fail; there is no common normalized plasma fetch schema to rely on.

## Configuration plot output

`l1obs.viz.configuration.plot_configuration(configuration, center="centroid", output=None)`
accepts a `SpacecraftConfiguration` and returns a Matplotlib figure with three
coordinate projections. Only centroid centering is implemented. Pass a string
or `Path` to save an image; plotting coordinates are centroid-relative in
thousands of kilometres.

```python
import matplotlib.pyplot as plt
from l1obs.proc.ephemeris import get_positions_at_time
from l1obs.viz.configuration import plot_configuration

configuration = get_positions_at_time(
    "2026-09-01T00:00:00Z", spacecraft=["Wind", "ACE"]
)
figure = plot_configuration(configuration, output="output/configuration.png")
plt.close(figure)
```

## HDF5 export

`l1obs.io.save.save_hdf5(df, outpath, key="l1obs")` writes a DataFrame to a
compressed HDF5 table, creates parent directories, and returns `None`. Supply a
`Path` output; an existing file is overwritten. Install PyTables with
`python -m pip install tables` first. This small example uses synthetic data:

```python
from pathlib import Path
import pandas as pd
from l1obs.io.save import save_hdf5

df = pd.DataFrame(
    {"b_mag": [5.0]}, index=pd.to_datetime(["2026-09-01T00:00:00Z"])
)
save_hdf5(df, Path("output/example.h5"))
```

## CDF export: incomplete

`l1obs.io.save.save_cdf(df, outpath)` is intended to export a DataFrame with a
datetime index to NASA CDF, with a TT2000 epoch and numeric data variables,
returning `None`. It currently has broken datetime conversion and missing CDF
writer imports, so it is **not a working export interface**. Invocation syntax
only, using a DataFrame such as `df` above:

```python
from pathlib import Path
from l1obs.io.save import save_cdf

# Incomplete implementation; do not run as a working export example.
# save_cdf(df, Path("output/example.cdf"))
```

## CLI commands

### `l1obs positions`

Supply `--time` to retrieve all configured spacecraft at one epoch. Prints
positions and centroid offsets in km, reports partial failures on stderr, and
opens a plot or saves it with `--output`:

```bash
l1obs positions --time 2026-09-01T00:00:00Z --output output/configuration.png
```

### `l1obs timeseries`

Supply `--start` to request one hour of magnetic/plasma observations across the
configured datasets. The time is floored to the UTC hour. `--cachedir` defaults
to `./cache`; `--outdir` defaults to `./output`; `--force` bypasses caches.

```bash
l1obs timeseries --start 2026-09-01T00:15:00Z --cachedir cache --outdir output
```

On success, this example writes `output/L1_Observatory_20260901_00UT.h5` and
`output/L1_Observatory_20260901_00UT_timeseries.png`. The command resamples to
1 second. It currently depends on the legacy plasma fetch path and PyTables,
so successful execution is not guaranteed with the default installation.
There is no magnetic-only CLI option; use the Python fetch call above.

Both commands support `-v`/`--verbose`, `-vv`, and `--help`.
