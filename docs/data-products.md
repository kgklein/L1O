# Data Products

## Ephemeris time series

`fetch_ephemeris` returns a pandas DataFrame:

| Field | Meaning | Units |
| --- | --- | --- |
| `time` (index) | UTC sample timestamps | Datetime |
| `x_km`, `y_km`, `z_km` | Spacecraft XYZ position in the requested frame | km |

The default frame is GSE (geocentric solar ecliptic); other SSCWeb coordinate
frames can be requested. Sampling follows the archive response, without
resampling. Rows containing missing or infinite coordinates are removed; samples
are sorted and duplicate timestamps retain the first record. This path does not
implement the magnetic products' CDF metadata validity filtering.

## Common-time spacecraft configuration

`get_positions_at_time` returns a `SpacecraftConfiguration`:

| Attribute | Contents |
| --- | --- |
| `time` | One timezone-aware UTC Python datetime |
| `frame` | Requested coordinate frame, uppercase; default `GSE` |
| `positions` | DataFrame of successful spacecraft positions |
| `errors` | Mapping of unavailable spacecraft labels to error messages |

The `positions` table has these columns:

| Column | Meaning | Units |
| --- | --- | --- |
| `spacecraft` | Display label, such as `Wind` or `ACE` | — |
| `x_km`, `y_km`, `z_km` | Position evaluated at the common epoch | km |
| `sample_before`, `sample_after` | UTC archive timestamps used for interpolation | Datetime |

Positions are linearly interpolated between bracketing samples. At an exact
sample, both provenance timestamps equal that sample's time. There is no
extrapolation. Partial successes remain available with failures in `errors`;
if every spacecraft fails, retrieval raises `RuntimeError`.

The configuration plot subtracts the available spacecraft centroid and displays
three projections in units of `10^3` km. The returned position table itself
retains absolute positions in the requested frame.

## Science fetch result

`fetch_magnetic_field` and `fetch_cdaweb_dataset` return the same `FetchResult`:

| Attribute | Contents |
| --- | --- |
| `df` | Retrieved DataFrame; magnetic schema below |
| `dataset_id` | Requested CDAWeb dataset identifier or NOAA product identifier |
| `used_vars` | Logical-to-archive variable mapping for a fresh fetch; `{"_cached": "true"}` for cache hits |

## Magnetic-field time series

| Spacecraft | Dataset | Archive time | GSE vector | Magnitude | Native cadence |
| --- | --- | --- | --- | --- | --- |
| Wind | `WI_H0_MFI` | `Epoch3` | `B3GSE` | `B3F1` | 3 s |
| ACE | `AC_H3_MFI` | `Epoch` | `BGSEc` | `Magnitude` | 1 s |
| DSCOVR | `DSCOVR_H0_MAG` | `Epoch1` | `B1GSE` | `B1F1` | 1 s |
| IMAP | `IMAP_MAG_L2_NORM-GSE` | `epoch` | `b_gse` | `magnitude` | 0.5 s |
| SOLAR-1 | `sci_mag-l3_solar1` (NOAA/NCEI science-quality) | `time_sec` | `b_gse_sec` | `b_gse_sphr_sec[:, 0]` | 1 s |

All five magnetic fetches produce this common DataFrame layout:

| Field | Meaning | Units/frame |
| --- | --- | --- |
| `time` (index) | Original sample timestamps, timezone-aware UTC | Datetime |
| `bx_gse` | X magnetic-field component | nT, GSE |
| `by_gse` | Y magnetic-field component | nT, GSE |
| `bz_gse` | Z magnetic-field component | nT, GSE |
| `b_mag` | Archive-provided field magnitude | nT |

Magnitude is retained independently of the vector. In particular, Wind's `B3F1`
is an average of magnitudes and may differ from the magnitude of its averaged
vector. Native fetching does not interpolate, resample, or drop invalid records.

### Missing values and quality

CDF `FILLVAL` values, nonfinite values, and values outside metadata `VALIDMIN` or
`VALIDMAX` become NaNs. Boundary values remain valid; vector components are
filtered independently. If `FILLVAL` is absent, the conventional `-1e31` sentinel
is recognized at float32/float64 precision. Missing validity bounds produce a
warning and are not replaced with physical thresholds.

DSCOVR additionally requires `FLAG1 == 0`; IMAP requires `quality_flags == 0`.
Nonzero, missing, or invalid flags mask all four magnetic values at that
timestamp. Both missions use the same metadata validity and quality-filtering
path. The normalized DataFrame does not retain the flag column.

SOLAR-1 comes from NOAA/NCEI rather than CDAWeb. Its Cartesian GSE columns
come from the three components of `b_gse_sec`; `b_mag` is the first component
of `b_gse_sphr_sec`, not a recomputed vector norm. NetCDF `_FillValue`,
`missing_value`, `valid_min`, `valid_max`, and `valid_range` metadata are applied
before normalization, with `-9999.0` as the fallback fill sentinel. No physical
bounds are invented. Only valid `flags_summary == 0` is accepted; degraded (1),
bad (2), missing, and invalid quality values mask all four fields. The separate
`flags` bitmask does not impose an additional filter.

`time_sec` uses microseconds since 1958-01-01, already corrected for leap seconds.
The provider normalizes the descriptive units string for xarray's CF decoding;
it does not apply another leap-second offset. Timestamps represent the start of
each one-second averaging window. SOLAR-1 fetching uses `[start, end)` and retains
invalid rows and timestamp gaps, without interpolating. Every requested day must
have a discoverable daily file; missing days raise an error.

### Processing and CLI output

The hourly CLI prefixes fields by spacecraft, for example `WIND_bx_gse` and
`WIND_b_mag`. It also retains `WIND_Br`, `WIND_Bt`, `WIND_Bn`, and `WIND_Bmag`
for its existing plotting interface. The component conversion is pseudo-RTN:
`Br = -bx_gse`, `Bt = -by_gse`, `Bn = bz_gse`; `Bmag` copies the archive magnitude.
This is a sign conversion rather than a full spacecraft-specific RTN transform.

The CLI resamples magnetic data to a 1-second grid only within contiguous valid
runs, independently per column. Invalid samples and surrounding gaps remain
NaN, and endpoints are not extrapolated. The public processing call
`l1obs.proc.resample.resample_to_1s(df, t0, t1, preserve_nan_gaps=True)` enables
this policy on the interval `[t0, t1)`. Its default `preserve_nan_gaps=False`
interpolates through NaNs and fills endpoints; the plasma CLI path uses that
default.

SOLAR-1 joins the same pipeline with `SOLAR-1_` column prefixes.
IMAP uses `IMAP_` column prefixes. Its
half-second samples remain available through direct fetching; the CLI's
1-second output does not retain every native sample.

Plasma data retain dataset-specific source columns, with selected prefixed
columns added by the CLI when available. There is no universal normalized plasma
product or standard temperature unit in the current implementation. See
[Input and Output](io.md) for the plasma fetch limitation.

## Six-spacecraft comparison alignment

The [comparison plotting routine](io.md#six-spacecraft-magnetic-and-geometry-comparison)
accepts separate magnetic and position DataFrames for each of exactly six
spacecraft. Inputs are copied and sorted; datetime indices must be unique and
contain no NaT. Naive times are interpreted as UTC. Numeric infinities are treated
as missing data.

The six magnetic panels retain their own native timestamps. Unlike the existing
archive-magnitude pipeline, this plot computes `|B|` from the three displayed
components by default. A missing component makes the computed magnitude missing.
Explicit `magnitude_column="b_mag"` uses the archive scalar without filling its
NaNs from the vector. Every panel uses the same magnetic quantity colors/styles;
spacecraft label colors match the geometry tracks.

By default the displayed window is the intersection of magnetic time coverages.
An explicit `time_range` can override it, provided each spacecraft has magnetic
samples in that interval. Magnetic data are never interpolated. NaNs break lines,
and gaps longer than three times each input's median positive sample spacing
receive a plotting-only break. Gap estimation uses the full input, before
clipping; `magnetic_max_gap` and `position_max_gap` can override it with positive
durations. These heuristics cannot identify every missing sample in sparse or
irregular data; use explicit limits when the sampling characteristics are known.

Geometry uses the sorted union of ephemeris timestamps within the shared valid
position coverage and displayed window, including covered interval boundaries.
XYZ positions are linearly interpolated in time only between consecutive valid
records. Any invalid XYZ row or oversized timestamp interval splits an
interpolation run; no extrapolation occurs. Single-record runs contribute only
at their exact timestamps. Break samples are retained even when all spacecraft
have a missing-timestamp gap, so geometry tracks do not bridge those gaps.

At each common time, the mesocenter is the arithmetic mean of **all six** finite
XYZ positions. If any spacecraft lacks a valid position, the complete geometry
sample is masked for all spacecraft. The routine never substitutes a five-member
center. It raises an error if there is no common coverage or no simultaneously
valid geometry sample.

The three geometry panels show `Δx–Δy`, `Δx–Δz`, and `Δy–Δz` relative to that
instantaneous center, with equal aspect and common distance limits. Hollow
circles indicate each track's first valid sample and filled triangles its last;
the black plus marks the center. The actual geometry time span is stated on the
figure and can be shorter than the magnetic window. Both columns require one
common declared/metadata frame, with no coordinate transformation. Default
geometry units are km; offsets are not divided into thousands of kilometres as
in the separate configuration plot.
