# L1 Observatory (l1obs)

Fetch + unify solar wind data from L1 spacecraft for a selected hour.
Outputs:
- merged 1-second cadence dataset (HDF5)
- stacked timeseries plot

## Install

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

## Run with:

l1obs --start 2026-01-17T12:00:00Z
