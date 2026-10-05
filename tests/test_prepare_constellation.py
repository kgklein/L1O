"""Offline preparation tests using mocked archives and a local PRADAN fixture."""
from datetime import date
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from l1obs.cli import _load_constellation_manifest
from l1obs.fetch.models import FetchResult

spec = importlib.util.spec_from_file_location("prepare_constellation", Path(__file__).resolve().parents[1] / "scripts" / "prepare_constellation.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PrepareConstellationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.output = self.cache / "constellation" / "2026-04-01"
        self.day = date(2026, 4, 1)
        self.start = pd.Timestamp("2026-04-01T00:00:00Z")
        times = pd.date_range(self.start, periods=3, freq="10s")
        ds = xr.Dataset({"time": ("utc_time", times.asi8 / 1e9, {"units": "Seconds (UNIX time)"}),
                         "x_gse": ("Bx", [100., -9999., 102.], {"units": "kilometers - km"}),
                         "y_gse": ("By", [200., 201., 202.], {"units": "kilometers - km"}),
                         "z_gse": ("Bz", [300., 301., 302.], {"units": "kilometers - km"}),
                         "Quality_flag_10s_data": ("utc_time", [0., 1., 1.])},
                        attrs={"Fill_value": "-9999.0"})
        self.local = self.cache / "L2_AL1_MAG_20260401_V00.nc"
        ds.to_netcdf(self.local, engine="netcdf4")
        self.mag = pd.DataFrame({"bx_gse": [3., 4., np.nan], "by_gse": [4., 4., np.nan],
                                 "bz_gse": [0., 0., np.nan], "b_mag": [20., 21., np.nan]}, index=times.rename("time"))
        self.positions = pd.DataFrame({"x_km": [100., 101., 102.], "y_km": [200., 201., 202.],
                                       "z_km": [300., 301., 302.]}, index=times.rename("time"))
        transport = patch("socket.socket.connect", side_effect=AssertionError("Unexpected network call"))
        transport.start()
        self.addCleanup(transport.stop)

    def test_prepares_six_pairs_and_reuses_them_offline(self):
        with patch.object(prepare, "fetch_magnetic_field", return_value=FetchResult(self.mag, "test", {})) as magnetic, \
             patch.object(prepare, "fetch_ephemeris", return_value=self.positions) as positions:
            manifest = prepare.prepare_day(self.day, self.cache, self.output)
        self.assertEqual(magnetic.call_count, 6)
        self.assertEqual(positions.call_count, 5)
        self.assertEqual(positions.call_args.args[1], self.start - pd.Timedelta(minutes=5))
        loaded = _load_constellation_manifest(manifest)
        self.assertEqual(list(loaded), [spec.label for spec in prepare.SPACECRAFT.values()])
        self.assertEqual(len(list(self.output.glob("*.parquet"))), 12)
        wind = loaded["Wind"]["magnetic"]
        pd.testing.assert_index_equal(wind.index, self.mag.index)
        self.assertEqual(wind.b_mag.iloc[0], 20)
        self.assertTrue(wind.iloc[-1].isna().all())
        self.assertEqual(loaded["Aditya-L1"]["positions"].x_km.iloc[0], 100)
        self.assertTrue(np.isnan(loaded["Aditya-L1"]["positions"].x_km.iloc[1]))
        with patch.object(prepare, "fetch_magnetic_field", side_effect=AssertionError("Unneeded fetch")), \
             patch.object(prepare, "fetch_ephemeris", side_effect=AssertionError("Unneeded fetch")):
            self.assertEqual(prepare.prepare_day(self.day, self.cache, self.output), manifest)

    def test_partial_failure_retains_products_without_manifest_and_retry_completes(self):
        def fetch(mission, *args, **kwargs):
            if mission == "ACE":
                raise RuntimeError("no archive coverage")
            return FetchResult(self.mag, "test", {})
        with patch.object(prepare, "fetch_magnetic_field", side_effect=fetch), \
             patch.object(prepare, "fetch_ephemeris", return_value=self.positions), \
             self.assertRaisesRegex(RuntimeError, "ACE magnetic.*no archive coverage"):
            prepare.prepare_day(self.day, self.cache, self.output)
        self.assertFalse((self.output / "inputs.json").exists())
        self.assertEqual(len(list(self.output.glob("*.parquet"))), 11)
        with patch.object(prepare, "fetch_magnetic_field", return_value=FetchResult(self.mag, "test", {})) as fetch, \
             patch.object(prepare, "fetch_ephemeris", side_effect=AssertionError("Already cached")):
            prepare.prepare_day(self.day, self.cache, self.output)
            self.assertEqual(fetch.call_count, 1)

    def test_force_refresh_and_missing_local_file_preflight(self):
        with patch.object(prepare, "fetch_magnetic_field", return_value=FetchResult(self.mag, "test", {})) as fetch, \
             patch.object(prepare, "fetch_ephemeris", return_value=self.positions):
            prepare.prepare_day(self.day, self.cache, self.output)
            fetch.reset_mock()
            prepare.prepare_day(self.day, self.cache, self.output, force=True)
            self.assertEqual(fetch.call_count, 6)
            self.assertTrue(fetch.call_args.kwargs["force"])
        with self.assertRaisesRegex(RuntimeError, "Aditya-L1 file is missing"):
            prepare.prepare_day(date(2026, 4, 2), self.cache, self.output)

    def test_rejects_wrong_units_and_bad_data(self):
        bad = self.positions.copy()
        bad.attrs["units"] = "m"
        with self.assertRaisesRegex(ValueError, "units must be km"):
            prepare._validate(bad, prepare.POSITIONS, "positions", self.start, self.start + pd.Timedelta(days=1))
        with self.assertRaisesRegex(ValueError, "no valid XYZ"):
            prepare._validate(self.positions * np.nan, prepare.POSITIONS, "positions", self.start, self.start + pd.Timedelta(days=1))
        bad = self.mag.copy()
        bad.index = pd.DatetimeIndex([self.start] * 3)
        with self.assertRaisesRegex(ValueError, "unique datetime"):
            prepare._validate(bad, prepare.MAGNETIC, "magnetic", self.start, self.start + pd.Timedelta(days=1))
