# L1 Observatory (l1obs)

Fetch, align, and visualize solar-wind observations and spacecraft positions
near Sun-Earth L1.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Hourly timeseries pipeline

Fetch, merge, save, and plot one hour of observations:

```bash
l1obs timeseries --start 2026-09-17T12:00:00Z
```

The start time is floored to the beginning of the UTC hour. Use
`l1obs timeseries --help` for cache and output options.

## Spacecraft configuration

Fetch SSCWeb ephemerides for the six configured spacecraft, interpolate them to
one common UTC epoch, print their GSE positions, and display a configuration
plot:

```bash
l1obs positions --time 2026-09-17T00:00:00Z
```

Save the three-panel X-Y, X-Z, and Y-Z plot instead of displaying it:

```bash
l1obs positions \
  --time 2026-09-01T00:00:00Z \
  --output l1_positions.png
```
