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

## CDAWeb fetch result

`fetch_cdaweb_dataset` returns `FetchResult`:

| Attribute | Contents |
| --- | --- |
| `df` | Retrieved DataFrame; magnetic schema below |
| `dataset_id` | Requested CDAWeb dataset identifier |
| `used_vars` | Logical-to-archive variable mapping for a fresh fetch; `{"_cached": "true"}` for cache hits |

## Magnetic-field time series

| Spacecraft | Dataset | Archive time | GSE vector | Magnitude | Native cadence |
| --- | --- | --- | --- | --- | --- |
| Wind | `WI_H0_MFI` | `Epoch3` | `B3GSE` | `B3F1` | 3 s |
| ACE | `AC_H3_MFI` | `Epoch` | `BGSEc` | `Magnitude` | 1 s |
| DSCOVR | `DSCOVR_H0_MAG` | `Epoch1` | `B1GSE` | `B1F1` | 1 s |

All three magnetic fetches produce this common DataFrame layout:

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

DSCOVR additionally requires `FLAG1 == 0`. Nonzero, missing, or invalid flags
mask all four magnetic values at that timestamp. The normalized DataFrame does
not retain the flag column.

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

Plasma data retain dataset-specific source columns, with selected prefixed
columns added by the CLI when available. There is no universal normalized plasma
product or standard temperature unit in the current implementation. See
[Input and Output](io.md) for the plasma fetch limitation.
