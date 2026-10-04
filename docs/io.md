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

## Shared magnetic-field retrieval

`l1obs.fetch.magnetic.fetch_magnetic_field(spacecraft, start, end, cache_dir, force=False)`
returns `FetchResult` containing native GSE fields and archive magnitude. Supply
one of `Wind`, `ACE`, `DSCOVR`, `IMAP`, or `SOLAR-1`, pandas timestamps, and a
`Path` cache directory. No resampling occurs. Naive requested times for SOLAR-1
are interpreted as UTC; explicit UTC timestamps are recommended for all missions.

```python
from pathlib import Path
import pandas as pd
from l1obs.fetch.magnetic import fetch_magnetic_field

result = fetch_magnetic_field(
    "SOLAR-1",
    pd.Timestamp("2026-04-22T00:00:00Z"),
    pd.Timestamp("2026-04-22T00:10:00Z"),
    Path("cache"),
)
print(result.df.head())
```

SOLAR-1 uses the NOAA/NCEI Space Weather Portal `/files` API to discover daily
science-quality `sci_mag-l3_solar1` NetCDF files. Returned URLs are used directly;
only overlapping days are downloaded, with the newest identifiable processing
revision selected. Multi-day requests are combined and trimmed to `[start, end)`.
Missing daily files raise an error listing the unavailable days.

NOAA raw daily files and versioned normalized parquet results are cached under
`cache_dir/ncei/sci_mag-l3_solar1/`. An interval cache hit needs no network;
`force=True` refreshes discovery and downloads. The other four missions retain
existing CDAWeb caching and fetching behavior. See [Data Products](data-products.md)
for time decoding, fill values, and quality filtering.

## CDAWeb science data and magnetic access

`l1obs.fetch.cdaweb.fetch_cdaweb_dataset(dataset_id, start, end, cache_dir, force=False)`
fetches a configured dataset. Supply UTC pandas timestamps and a `Path` cache
directory. Returns `FetchResult` with `df`, `dataset_id`, and `used_vars`. Requests
reuse parquet caches; `force=True` downloads again. Magnetic caches are versioned
to exclude older, unnormalized products.

Magnetic access uses this same call with `WI_H0_MFI`, `AC_H3_MFI`,
`DSCOVR_H0_MAG`, or `IMAP_MAG_L2_NORM-GSE`; fetching preserves native cadence
and returns normalized fields. IMAP retains 0.5-second samples, reads the
lowercase archive time variable `epoch`, and masks samples whose
`quality_flags` are nonzero, missing, or invalid. Its archive-provided
`magnitude` becomes `b_mag`.

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

## Six-spacecraft magnetic and geometry comparison

`l1obs.viz.constellation.plot_six_spacecraft_B_and_geometry` plots six native
magnetic time series beside three instantaneous-mesocenter geometry projections.
It takes exactly six named entries, each containing separate `magnetic` and
`positions` DataFrames with datetime indices. No retrieval occurs. L1O currently
fetches magnetic data for five missions; other inputs must already be available
from your own sources.

Given already-loaded `magnetic_by_spacecraft` and `positions_by_spacecraft`
dictionaries containing the same six names and GSE data:

```python
from pathlib import Path
import matplotlib.pyplot as plt
from l1obs.viz.constellation import plot_six_spacecraft_B_and_geometry

datasets = {
    name: {"magnetic": magnetic, "positions": positions_by_spacecraft[name]}
    for name, magnetic in magnetic_by_spacecraft.items()
}
figure = plot_six_spacecraft_B_and_geometry(
    datasets,
    coordinate_system="GSE",
    output_path=Path("output/constellation.png"),
    save_pdf=True,
)
plt.close(figure)
```

The result is an open Matplotlib figure with nine axes. This example saves a
300-dpi PNG and a sibling PDF. Without `output_path`, no file is written;
`show=True` displays the figure. Output paths must end in `.png`.

| Option | Behavior |
| --- | --- |
| `spacecraft_names` | Explicit ordering; must be a permutation of the six dataset names |
| `time_range=(start, end)` | UTC display interval, including endpoint samples; otherwise the shared magnetic time coverage |
| `B_components`, `position_components` | Three columns each; defaults are `bx_gse`, `by_gse`, `bz_gse` and `x_km`, `y_km`, `z_km` |
| `B_units`, `position_units` | Input unit declarations and labels; default nT and km, with no conversion |
| `coordinate_system` | Common frame declaration, required if any input lacks frame metadata |
| `common_B_ylim` | Shared magnetic limits by default; `False` autoscales each panel |
| `magnitude_column` | Default `None` computes the displayed vector norm; explicitly select `"b_mag"` for archive magnitude |
| `magnetic_max_gap`, `position_max_gap` | Positive timedelta-like overrides, e.g. `"5s"` or `"5min"`; default three times each series' median spacing |
| `colors` | Mapping from spacecraft names to colors; known names otherwise use repository colors |

The plot checks `DataFrame.attrs["frame"]` and `attrs["coordinate_system"]`,
when present, against the common frame declaration and each other. Unit metadata
may be a scalar `attrs["units"]` or a mapping from column names to units. Missing
metadata is supplied by your declarations; conflicting metadata raises an error.
`_gse` field columns require GSE, and `_km` position columns require km. The plot
does not transform frames or units. See [alignment and gap behavior](data-products.md#six-spacecraft-comparison-alignment).

### CLI: `l1obs constellation`

Use a local JSON manifest to call this plotter directly. The command reads files
without contacting CDAWeb or SSCWeb. For example, save `inputs.json` containing
exactly six unique spacecraft names:

```json
{
  "Wind":      {"magnetic": "wind_mag.parquet",   "positions": "wind_pos.parquet"},
  "ACE":       {"magnetic": "ace_mag.parquet",    "positions": "ace_pos.parquet"},
  "DSCOVR":    {"magnetic": "dscovr_mag.parquet", "positions": "dscovr_pos.parquet"},
  "Aditya-L1": {"magnetic": "aditya_mag.csv",     "positions": "aditya_pos.csv"},
  "IMAP":      {"magnetic": "imap_mag.parquet",   "positions": "imap_pos.parquet"},
  "SOLAR-1":   {"magnetic": "solar1_mag.csv",     "positions": "solar1_pos.csv"}
}
```

These are example local filenames, not a claim that L1O can fetch magnetic data
for all six missions. Supply your own loaded data for the other spacecraft.
File paths are resolved relative to the manifest directory; absolute paths are
also accepted. Manifest order determines the panel order.

Parquet files must contain a datetime index or a `time` column. CSV files require
a `time` column containing parseable timestamps. Magnetic and position files
retain independent cadences. Existing DataFrames can be written with
`frame.rename_axis("time").to_parquet(path)` or
`frame.rename_axis("time").to_csv(path)`; the default field and position columns
are listed above. CSV does not preserve frame/unit metadata, so declare the
frame explicitly and supply the correct units.

```bash
l1obs constellation --manifest inputs.json --coordinate-system GSE \
  --output output/constellation.png --pdf
```

The default output is `./output/constellation.png`. Add `--show` for interactive
display. To limit the interval, supply both `--start` and `--end`; times are
interpreted as UTC and are **not** floored to an hour.

The command also accepts `--spacecraft-order` followed by all six names,
`--b-components` and `--position-components` followed by three columns each,
`--b-units`, `--position-units`, `--independent-b-ylim`, `--magnitude-column`,
`--magnetic-max-gap`, and `--position-max-gap`. Frame and unit declarations do
not perform conversions. The Python interface remains available for custom
color overrides. Run `l1obs constellation --help` for all options, including
`-v`/`--verbose` and `-vv`.
