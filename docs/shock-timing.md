# Native-data shock timing (Stages 1A–1C)

L1O can inspect an event and invert **explicitly accepted** multispacecraft
crossings for a planar, constant-speed front. The April 1, 2026 example starts
with **no accepted arrivals**. Detector output is provisional. No measured shock
normal or speed is supplied with this implementation.

## Reproduce the April 1 inspection

From the repository root, after installing L1O:

```bash
l1obs shock inspect \
  --event examples/events/20260401_shock.json \
  --manifest cache/constellation/2026-04-01/inputs.json -v
```

This is entirely offline. If those inputs are absent, obtain the original files
with the existing magnetic/ephemeris interfaces first. The existing
`scripts/prepare_constellation.py --help` describes the daily manifest preparation
workflow; Aditya-L1 requires an already downloaded PRADAN daily file. Do not feed
the hourly interpolated 1-second HDF5 science product into this analysis.

The manifest is the same mapping used by `l1obs constellation`:

```json
{
  "Wind": {"magnetic": "wind_magnetic.parquet", "positions": "wind_positions.csv"},
  "ACE": {"magnetic": "ace_magnetic.csv", "positions": "ace_positions.parquet"}
}
```

Include all event members (six for April 1). Paths in a manifest are relative to
that manifest. CSV tables need a `time` column; Parquet supports a datetime index
or `time` column. Magnetic columns are `bx_gse`, `by_gse`, `bz_gse`, and optional
`b_mag` in nT. Positions are `x_km`, `y_km`, `z_km` in GSE km. A supplied `b_mag`
is preserved independently of component validity; its NaNs are never filled from
the vector. If absent, the vector norm is calculated, requiring all components.
Negative magnitudes become NaN. Original data files are never changed.

Normalized inputs must already carry the quality masks and fill-value treatment
of the existing mission loaders. This avoids a competing interpretation of raw
PRADAN/NOAA flags. The event explicitly asserts `native_cadence: true` and
`quality_masks_applied: true`, plus GSE/nT/km units. Contradictory file metadata,
duplicate timestamps, missing members, or a declared resampled product are
errors. A normalized CSV cannot prove its retrieval history: these assertions
are the analyst's responsibility. The configured mission dataset identifier is
recorded as configured, **not independently verified**. Input paths, attributes,
and SHA-256 digests record the actual files used.

## Event definition and manual validation

`examples/events/20260401_shock.json` defines 10:45–12:15 UTC inspection and broad
11:10–11:40 search windows. The windows are starting points, not observations.
Event manifest/output paths resolve relative to the event JSON; CLI overrides
resolve relative to the working directory. Naive timestamps are interpreted as
UTC; timezone-aware inputs are converted to UTC. Date boundaries are supported.
Leap-second labels themselves are not represented by pandas timestamps.

Inspect `shock_inspection`, `shock_components`, and `shock_zoom`. The common-axis
inspection uses original epochs; the zoom gives each row its own physical-time
window. `--zoom-seconds 30` sets its half-width. `shock_relative` plots `t-t_i`
only for comparison and never changes the scientific timestamps. Omit it with
`--no-relative`. All plots are saved as PNG and PDF. Spacecraft labels and
geometry use the stable project colors; Bx, By, Bz use the existing blue solid,
red dashed, green dotted styles, with magnitude black.

Copy or edit the event and enter physical crossing epochs in `arrivals`. For
each spacecraft use this structure (replace the placeholder with your observed
UTC epoch; the following is a schema illustration, not a usable observation):

```json
{
  "ACE": {
    "time": "<validated UTC crossing>",
    "uncertainty_s": 2.0,
    "accepted": true,
    "method": "manual_validated",
    "notes": "Describe ramp choice, source epoch convention, and clock limitations."
  }
}
```

Use `manual_selected` for an independently chosen time or `manual_validated` for
a reviewed proposal. A time alone never implies acceptance. A future inspection
retains configured times and uncertainties verbatim and does not write the event
file. `arrivals.csv` records the manual/proposal state, cadence, ramp duration,
uncertainty, notes, and timestamp convention. Fit requires every requested event
member to have an explicitly accepted manual arrival, including all six in the
April 1 event. There is no automatic fallback to a smaller constellation.

```bash
# Run only after accepting all six physical arrivals in the event:
l1obs shock fit \
  --event examples/events/20260401_shock.json \
  --manifest cache/constellation/2026-04-01/inputs.json -vv

# Explicit subset, with at least four geometrically independent spacecraft:
l1obs shock fit --event examples/events/20260401_shock.json \
  --spacecraft Wind ACE DSCOVR IMAP SOLAR-1
```

Four accepted spacecraft give zero residual degrees of freedom. Leave-one-out
requires at least five input spacecraft. A subset with singular geometry is
reported as failed, not substituted with another subset. All commands preserve
the existing `timeseries`, `positions`, and `constellation` workflows.

## Candidate detection and timing precision

The detector works independently on each magnetic magnitude:

1. Restrict to its search window and split at every invalid magnitude or
   timestamp separation exceeding `max_gap_s` (default three median spacings).
2. At consecutive-sample midpoints compare upstream and downstream windows of
   `window_s` each (default 60 s). Each side needs three samples and at least
   60% of the requested temporal support.
3. Require a positive downstream-minus-upstream median jump above `min_jump_nt`
   and a jump-to-robust-scatter ratio above `min_significance`. Scatter is the
   larger side's 1.4826 MAD, bounded by `noise_floor_nt` (default 0.05 nT).
   This ratio is a heuristic, **not a calibrated detection probability**.
4. Rank by jump amplitude, using adjacent positive slope to break ties. Find the
   nearest ascending 50% threshold crossing of those plateau medians by linear
   interpolation between two consecutive native samples. Infer 10–90% width in
   the same contiguous run. Unresolved widths are bounded below by cadence and
   the local sample spacing. Inconsistent threshold ordering is a complex ramp.
5. Suppress duplicate candidates within one comparison-window duration, retain
   up to five, and mark ambiguity when a second jump has at least
   `ambiguity_ratio` (default 0.8) of the strongest amplitude, or the best ramp is
   complex. Nearby discontinuities within the suppression duration may be
   inseparable; change the window and inspect the original profile.

All qualifying candidates and their statistics remain in `inspection.json` (and
the fit report). Dotted markers show alternatives; the chosen proposal is dashed,
accepted crossings solid, and shading is ±1 sigma. A missing/no-candidate status
is explicit. No smoothing or resampling is currently performed. Missing
*timestamp* gaps smaller than the configured gap limit cannot be distinguished
from legitimate nonuniform sampling; choose an instrument-appropriate limit.

Proposal sigma is the maximum of half native cadence, half the bounded ramp
width, and the explicitly supplied `timestamp_uncertainty_s`. This is a
conservative, interpretable starting estimate, not a calibrated error model for
all physical ramps. Manual sigmas must reflect bin averaging, ramp identification,
clock knowledge, and any uncertain bin-reference offset. Values below half
native/local spacing or the timestamp floor require a nonempty
`precision_justification`. A justification records the claim; it does not verify
it. An accepted ramp duration can be entered as `ramp_duration_s`. Missing
uncertainties allow OLS, but partial uncertainty vectors are rejected.

Detector epochs are written to millisecond resolution for serialization only;
no millisecond physical precision is claimed. Ramp localization is not determined
by an interpolated 1-second science grid. The native Wind (~3 s), ACE (~1 s),
DSCOVR (~1 s), Aditya-L1 (10 s averaged), IMAP (~0.5 s), and SOLAR-1 (1 s averaged)
products retain their actual sample times and NaNs.

### Timestamp conventions

The repository's loaders already decode mission time variables; this analysis
applies no further leap-second conversion. The source checks for this stage are:

| Source | Verified information | Remaining uncertainty |
| --- | --- | --- |
| Wind `Epoch3`, ACE `Epoch`, DSCOVR `Epoch1`, IMAP `epoch` | Existing loaders use those archive epoch variables at native product cadence | Bin reference and absolute clock accuracy have not been verified by this implementation |
| Aditya-L1 PRADAN `time` | Local NetCDF says Unix UTC seconds and 10-second bin width; [ISSDC MAG FAQ](https://pradan1.issdc.gov.in/al1/faq.xhtml) confirms L2 time averaging | Start versus center and clock uncertainty remain unresolved; do not infer this from Unix encoding |
| SOLAR-1 `time_sec` | Local NOAA source NetCDF `comments` says each timestamp labels the start of its one-second averaging window; archive counts are already leap-second corrected | Clock error is not established by the bin label |

The [Wind archive notes](https://cdaweb.gsfc.nasa.gov/misc/NotesW.html#WI_H0_MFI)
and [ACE archive notes](https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H3_MFI)
are product references; cadence alone does not establish an epoch's bin reference.

Optional `inputs.spacecraft_metadata` holds `timestamp_convention`,
`timestamp_offset_s`, and `timestamp_uncertainty_s` per spacecraft. The example
explicitly sets SOLAR-1's proposal offset to **+0.5 s**, placing the averaged
measurement at its bin center for detection. Original timestamps remain intact.
Other offsets default to zero, with unresolved conventions clearly recorded.
An offset is a deterministic proposed-time correction; a sigma is stochastic
uncertainty. A manual `arrivals.time` is already a physical crossing epoch and
is **never shifted again**. Record any manually applied correction in notes.
This distinction is essential when comparing different averaging conventions.

## Crossing-epoch geometry and GSE approximation

For each accepted `t_i`, `positions_at_arrivals` calls the existing
`interpolate_position` on the consecutive bracketing observations. An exact
finite sample is used directly; otherwise both neighbors must be finite. No
extrapolation, interpolation across an invalid row, or bracket longer than
`inputs.ephemeris_max_gap_s` is allowed. The default 900 s accommodates the
12-minute ephemerides present in some local April 1 files. Analysts should
review whether linear interpolation on that interval meets their positional
accuracy requirement. It is a maximum gap policy, not an error bound.

The output records the actual crossing epoch, before/after samples, bracket gap,
method, frame, and units for each spacecraft. Original Earth-centered positions
remain available; centroid-relative coordinates are for conditioning and figures.
During inspection, proposed epochs are used, with the inspection midpoint only
for an unselected spacecraft, clearly identified in `inspection.json`. Missing
inspection positions are recorded. A fit fails if any required position is
unavailable. Upstream SSCWeb retrieval drops invalid coordinates, so a missing
archive record is identifiable only through the resulting gap bound.

GSE's rotating axes are approximated as a common fixed frame across the crossing
interval. No RTN transform occurs. A diagnostic uses annual angular rate
`2π/(365.25 day)` and estimates `max(|r_i|) ω Δt` as a conservative displacement
scale, then divides by fitted shock speed for a timing scale. This estimate
uses Earth-centered radii; it is neither a full time-dependent transformation nor
an uncertainty correction. Sub-second input sigmas always trigger an explicit
frame/timestamp review flag. For higher precision, transform each position into
a single epoch's consistent frame and assess ephemeris/clock errors in future
work. This stage keeps GSE throughout as specified.

## Planar timing equations and solution

For spacecraft crossing at physical epoch `t_i` and its position `r_i(t_i)`:

```text
t_i - t_ref = a + q · (r_i - r_ref) + epsilon_i
q = n / V_sh
V_sh = 1 / |q|                  [km/s]
n = q / |q|                    [GSE unit vector]
```

`n` points in the **propagation direction**: arrivals increase with position
along `n`. The fitted sign is preserved. No sunward/antisunward assumption or
post-fit sign flip is imposed. Positions are km; slowness is s/km. The fitted speed is kinematic in the
approximately frozen Earth-centered GSE frame, not a speed in the upstream
plasma rest frame.

The reference time is the earliest input arrival; the reference position is the
arithmetic constellation centroid at the individual crossing epochs. Relative
times are formed before floating-point linear algebra. The full affine design
has one intercept and three spatial columns; no spacecraft is treated as an
error-free reference. Centering does not remove the fitted intercept. No
reference-subtracted correlated measurement errors are introduced.

NumPy SVD solves the least-squares problem. All spatial columns are divided by
one common RMS spatial scale, avoiding unit-driven design conditioning while
preserving the actual anisotropy. The unweighted centered geometry SVD must have
rank three. Both the geometry and weighted/scaled affine design must have
condition number ≤ `fit.condition_limit` (default 1e6); rank uses the usual
machine-epsilon × matrix-size × largest-singular-value tolerance. Conditions
above 1000 are flagged as weak even when below the rejection threshold. These
are configurable numerical policies, not a complete scientific geometry test.
Zero or numerically unresolved slowness is an error, not infinite speed.

`fit.method: "auto"` uses weighted least squares when complete uncertainties are
available and OLS otherwise. `weighted` requires uncertainties/covariance;
`unweighted` explicitly selects OLS. With independent errors, minimize
`sum((epsilon_i/sigma_i)^2)`. Finite, strictly positive sigmas are mandatory.
An optional positive-definite symmetric `fit.covariance_s2` matrix in event
spacecraft order enables Gaussian correlated-error GLS. Cholesky whitening is
applied to both the affine design and times. If sigmas are also given, they must
match its diagonal. Explicit subsets select the corresponding covariance block.
Supplied absolute errors are never rescaled to force reduced chi-square to one.
For explicitly unweighted fits with heterogeneous/correlated known errors, the
weighted residual statistic is flagged as uncalibrated for chi-square inference.

The parameter covariance is in `[a, qx, qy, qz]` units. It uses the known noise
model (sandwich covariance also for explicitly unweighted fits) or, for OLS
without supplied errors and positive dof, residual variance. It is not the sole
basis for the reported nonlinear speed/direction intervals.

## Residual diagnostics and sensitivity

Predicted physical epochs are `t_ref + a + q·(r_i-r_ref)`; residuals are
**observed minus predicted**, in seconds. `timing_residuals.csv` contains physical
timestamps and relative offsets, preserving the centered calculation. Reported
statistics include RMS and maximum absolute residual, chi-square when errors are
supplied, `N-4` degrees of freedom, and reduced chi-square only for positive dof.

Six members supply **two** residual dof, so goodness-of-fit inference is limited.
Five-member leave-one-out fits have **one**; four-member fits cannot test the
model through residuals. The implementation flags reduced chi-square >4 or an
individual residual >3 supplied sigmas as review heuristics, not calibrated
hypothesis-test thresholds. No p-value or model-validation claim is made.

Each leave-one-out row contains the omitted member, signed speed change,
propagation normal, angular change from the full solution, geometry/design
conditioning, subset residuals, dof, chi-square, and failure status if needed.
These fits use unchanged accepted epochs and their individual positions.
A single influential member can be essential for geometric independence rather
than incorrect; interpret sensitivity together with geometry and uncertainties.

## Monte Carlo propagation

The default uses 2000 draws with an explicit deterministic seed (20260401 in the
example). Each draw perturbs all accepted physical times by their supplied
independent Gaussian 1-sigma errors, or by the supplied covariance's Gaussian
model. With `recompute_positions: true` (default), every position is re-evaluated
at its perturbed crossing time using the same bracketing/gap policy. Missing
support produces a failed draw, never extrapolation. With
`recompute_positions: false`, positions remain fixed and this approximation is
recorded. Missing ephemerides never cause a silent fixed-position fallback.

Each realization records speed, normal, conditioning, residuals, and weighted
fit diagnostics, or its explicit error. `monte_carlo_draws.csv` includes failures.
The JSON reports success/failure counts and reasons. Intervals condition on
successful draws; any failure flags that conditioning, and success below 80%
is marked unstable. Even smaller failure fractions can bias an interval near a
coverage boundary and warrant checking the ephemeris extent.

Speed summaries are the median and central percentile intervals [16,84] and
[2.5,97.5]. The central direction is the normalized mean of successful unit
normals. The 68% and 95% angular radii contain that fraction of directions about
the center, using `acos(clip(n_draw·n_center,-1,1))`. These are directional caps,
not independent Cartesian-component error bars. Resultant length, angular radii
about the nominal fit, weak-conditioning counts, and hemisphere reversals are
also reported. A mean resultant length below 0.9 or a 95% radius above 45° marks
the solution unstable; one confidence cone is then insufficient. These quality
thresholds are documented heuristics.

Monte Carlo does not repair a biased arrival choice or make a poorly constrained
fit precise. It assumes the supplied error distribution is meaningful. This
stage excludes ephemeris uncertainty, unknown shared clock bias unless modeled
in covariance, asymmetric ramp uncertainty, solar-wind evolution, and selection
bias from comparing several discontinuities. With no timing errors supplied,
Monte Carlo is unavailable and explicitly noted.

## Products and Python interfaces

Outputs default to `output/shocks/20260401_shock/`:

| Product | Contents |
| --- | --- |
| `arrivals.csv` | Selected/proposed state and native timing metadata |
| `inspection.json` | Inspection event, candidate alternatives, geometry status, inputs, software |
| `shock_timing.json` | Fit, uncertainty summary, geometry, assumptions, sensitivity, flags |
| `timing_residuals.csv` | Observed/predicted physical epochs and residuals |
| `leave_one_out.csv` | Subset solutions and failures |
| `monte_carlo_draws.csv` | Successful and failed uncertainty realizations |
| `shock_inspection.png/.pdf` | Stacked native magnitudes and selected crossings |
| `shock_components.png/.pdf` | Magnitudes plus GSE components |
| `shock_zoom.png/.pdf` | Per-spacecraft ramp zooms |
| `shock_relative.png/.pdf` | Optional relative-crossing profiles |
| `shock_geometry.png/.pdf` | Equal-scale 3D and GSE projections; front/normal after fit |
| `timing_diagnostics.png/.pdf` | Residual errors and leave-one-out speeds/angles |

The plane is placed at the fitted reference epoch, satisfying
`q·(r-r_ref)=-a`; points show positions at their own crossings, so they should
**not all lie on that instantaneous plane**. Geometry is centroid-relative in
km, with equal physical scales. Reports include the event, analysis windows,
file hashes and metadata, package versions, Git revision/dirty state, and Python
source hashes (including uncommitted implementation files).

Output filenames are reusable: a run replaces its generated products, but never
its input event/tables. On fit failure, `shock_timing.json` explicitly records
`status: "failed"` and no fitted solution. Older sibling images/CSVs may still
exist; the current JSON status and input hashes govern interpretation. Use a new
`--outdir` for each scientific revision when preserving prior results.

Public imports from `l1obs.analysis.shocks`:

- `Event`, `EventData`, `load_event`, `load_event_data`.
- `position_at`, `positions_at_arrivals`.
- `DetectionOptions`, `DetectionResult`, `detect_arrivals`, `build_arrival_table`,
  `accepted_arrivals`.
- `TimingFit`, `fit_timing`, `leave_one_out`, `angular_separation_deg`.
- `monte_carlo_timing`.

`l1obs.analysis.shocks.pipeline.run_event` is the offline orchestration API.
Numerical modules do not import plotting/CLI code. Existing NumPy, pandas,
Matplotlib and Parquet dependencies suffice; no new runtime dependency is added.
Python 3.10+ is supported.

## Validation and scientific scope

Run the repository's unittest suite:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

Synthetic tests cover noiseless recovery, realistic timing noise, differential
weights, equal-weight OLS/WLS agreement, sign/units, frame/time translations,
singular/nearly coplanar geometry, invalid inputs, four-member fits, corrupted
arrival sensitivity, Gaussian covariance, deterministic Monte Carlo, moving
position propagation, explicit failed draws, no ephemeris extrapolation,
magnetic invalid/gap boundaries, date boundaries, manual acceptance persistence,
and offline CSV/Parquet CLI products. A controlled 60-event Gaussian experiment
checks empirical 68%/95% speed and directional-cap coverage using 180 draws per
event with broad binomial tolerances. It is a regression check under the stated
noise model, not validation of observational error distributions.

The available April 1 local inputs have been inspected through this workflow;
all six native series and ephemerides load and diagnostic plots are produced.
The checked-in example deliberately retains empty `arrivals`. Human validation
of the six ramp choices, epoch conventions, and errors is still required before
an observational fit. No synthetic fit is reported as a measurement. In
particular the detector picks positive enhancements independently; identifying
the same physical front at all six spacecraft remains a scientific judgment.

Residuals test compatibility with one planar constant-speed front. A good fit
does not prove planarity, and a poor fit does not uniquely diagnose curvature,
rippling, acceleration, front evolution, timing bias, or misidentification.
Magnetic coplanarity, plasma moments, jump conditions, Rankine–Hugoniot analysis,
Mach numbers, curvature reconstruction, and automatic shock classification are
outside Stages 1A–1C.
